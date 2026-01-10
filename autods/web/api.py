import asyncio
import concurrent.futures
import hashlib
import io
import json
import logging
import os
import queue
import threading
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from langchain_core.messages import (
    AIMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    ToolMessage,
)
from pydantic import BaseModel

from autods.runtime.runner import AgentRunner
from autods.sessions import SessionMetadata, SessionNotFoundError, SessionService
from autods.utils.config import load_config

from ..web.loggers import Tracer

logger = logging.getLogger(__name__)
PROJECTS_ROOT = Path(os.environ.get("PROJECTS_ROOT", "/tmp/autods/projects"))
PROJECTS_ROOT.mkdir(parents=True, exist_ok=True)
ARTIFACT_TREE_MAX_DEPTH = int(os.environ.get("ARTIFACT_TREE_MAX_DEPTH", "5"))
ARTIFACT_TREE_MAX_ITEMS = int(os.environ.get("ARTIFACT_TREE_MAX_ITEMS", "10000"))
SNAPSHOT_ROOT = PROJECTS_ROOT / "_snapshots"
SNAPSHOT_ROOT.mkdir(parents=True, exist_ok=True)
ALLOWED_UPLOAD_EXTENSIONS = {
    ".csv",
    ".tsv",
    ".parquet",
    ".json",
    ".jsonl",
    ".md",
    ".txt",
    ".rst",
    ".yaml",
    ".yml",
    ".toml",
    ".py",
    ".ipynb",
}

from autods.agents.autods import AutoDSAgent


class TaskRequest(BaseModel):
    task: str
    provider: Optional[str] = None
    model: Optional[str] = None
    model_base_url: Optional[str] = None
    api_key: Optional[str] = None
    max_steps: Optional[int] = None
    project_path: Optional[str] = None
    config_file: Optional[str] = None
    session_id: Optional[str] = None


class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None


class SessionResponse(BaseModel):
    id: str
    created_at: datetime
    updated_at: datetime


class TaskResponse(BaseModel):
    session_id: str
    status: str
    message: Optional[str] = None


class ConfigUpdateRequest(BaseModel):
    config: Dict[str, Any]


class AddDatasetRequest(BaseModel):
    url: str


class InstallLibrariesRequest(BaseModel):
    libraries: List[str]


class WebSocketManager:
    def __init__(self):
        self.active_connections: Dict[str, set[WebSocket]] = {}
        self.history: Dict[str, List[str]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        async with self._lock:
            self.active_connections.setdefault(session_id, set()).add(websocket)
            if session_id in self.history:
                for msg in self.history[session_id]:
                    await websocket.send_text(msg)
        logger.info(
            "WebSocket connected for session %s (total %d)",
            session_id,
            len(self.active_connections[session_id]),
        )

    async def disconnect(self, session_id: str, websocket: Optional[WebSocket] = None):
        async with self._lock:
            connections = self.active_connections.get(session_id)
            if not connections:
                return
            if websocket:
                connections.discard(websocket)
            else:
                for ws in list(connections):
                    try:
                        await ws.close()
                    except RuntimeError:
                        pass
                connections.clear()
            if not connections:
                self.active_connections.pop(session_id, None)
        logger.info("WebSocket disconnected for session %s", session_id)

    async def send_message(self, session_id: str, message: str):
        async with self._lock:
            if session_id not in self.history:
                self.history[session_id] = []
            self.history[session_id].append(message)
            if len(self.history[session_id]) > 100000:
                self.history[session_id] = self.history[session_id][-50000:]
            connections = list(self.active_connections.get(session_id, ()))
            # Send while holding lock to prevent message interleaving
            failed: list[WebSocket] = []
            for websocket in connections:
                try:
                    await websocket.send_text(message)
                except Exception as exc:
                    logger.error("WebSocket send failed (%s): %s", session_id, exc)
                    failed.append(websocket)
        # Disconnect failed sockets outside lock to avoid deadlock
        for ws in failed:
            await self.disconnect(session_id, ws)

    async def close_session(
        self, session_id: str, message: Optional[str] = None, code: int = 1000
    ):
        if message:
            await self.send_message(session_id, message)
        async with self._lock:
            connections = list(self.active_connections.pop(session_id, ()))
        for websocket in connections:
            await websocket.close(code=code)
        if connections:
            logger.info(
                "Closed %d WebSocket connection(s) for %s", len(connections), session_id
            )


class CancellationRegistry:
    """Thread-safe registry for managing session cancellation."""

    def __init__(self):
        self._lock = threading.Lock()
        self._events: Dict[str, threading.Event] = {}
        self._runners: Dict[str, "AgentRunner"] = {}

    def register(self, session_id: str, runner: "AgentRunner") -> threading.Event:
        """Register a new session and return its cancellation event."""
        with self._lock:
            event = threading.Event()
            self._events[session_id] = event
            self._runners[session_id] = runner
            return event

    def cancel(self, session_id: str) -> bool:
        """Signal cancellation for a session. Returns True if session was found."""
        with self._lock:
            if session_id not in self._events:
                return False
            self._events[session_id].set()
            return True

    def is_cancelled(self, session_id: str) -> bool:
        """Check if session was cancelled."""
        with self._lock:
            event = self._events.get(session_id)
            return event.is_set() if event else False

    def unregister(self, session_id: str) -> None:
        """Cleanup when session completes."""
        with self._lock:
            self._events.pop(session_id, None)
            self._runners.pop(session_id, None)


# Global cancellation registry
cancellation_registry = CancellationRegistry()


class APIMessageStreamPrinter:
    def __init__(
        self,
        session_id: str,
        websocket_manager: WebSocketManager,
        loop: asyncio.AbstractEventLoop,
    ):
        self.session_id = session_id
        self.websocket_manager = websocket_manager
        self.loop = loop

    async def print_chunk_callback(self, chunk_type: str, data: Any):
        message = json.dumps(
            {
                "type": chunk_type,
                "data": str(data),
                "timestamp": datetime.now().isoformat(),
            }
        )
        asyncio.run_coroutine_threadsafe(
            self.websocket_manager.send_message(self.session_id, message), self.loop
        )


def get_session_workspace(session_id: str, *, create: bool = True) -> Path:
    workspace = PROJECTS_ROOT / session_id
    if create:
        workspace.mkdir(parents=True, exist_ok=True)
    return workspace


def sanitize_relative_path(path: str) -> str:
    return Path(path.strip()).as_posix()


def _artifacts_root(session_id: str) -> Path:
    return get_session_workspace(session_id, create=False)


def create_app(agent_options: Optional[dict[str, Any]] = None) -> FastAPI:
    app = FastAPI(title="AutoDS API", version="0.1.0")
    default_agent_options = agent_options or {}

    def _merge_agent_options(task_request: TaskRequest) -> dict[str, Any]:
        merged = dict(default_agent_options)
        for field in (
            "provider",
            "model",
            "model_base_url",
            "api_key",
            "max_steps",
            "config_file",
            "project_path",
        ):
            value = getattr(task_request, field, None)
            if value is not None:
                merged[field] = value
        return merged

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    manager = WebSocketManager()

    class ThreadSafePrinter:
        def __init__(
            self,
            session_id: str,
            ws_manager: WebSocketManager,
            main_loop: asyncio.AbstractEventLoop,
        ):
            self.session_id = session_id
            self.ws_manager = ws_manager
            self.main_loop = main_loop

        async def print_chunk_callback(self, chunk_type: str, data: Any):
            message = json.dumps(
                {
                    "type": chunk_type,
                    "data": str(data),
                    "timestamp": datetime.now().isoformat(),
                }
            )
            asyncio.run_coroutine_threadsafe(
                self.ws_manager.send_message(self.session_id, message), self.main_loop
            )

    # def run_agent_sync_wrapper(
    #         task_request: TaskRequest,
    #         session: SessionMetadata,
    #         runner: AgentRunner,
    #         ws_manager: WebSocketManager,
    #         session_service: SessionService,
    #         main_loop: asyncio.AbstractEventLoop
    # ):
    #     local_loop = asyncio.new_event_loop()
    #     asyncio.set_event_loop(local_loop)
    #     thread_safe_printer = ThreadSafePrinter(session.id, ws_manager, main_loop)
    #
    #     async def _async_job():
    #         try:
    #             await runner.astream(
    #                 task_request.task,
    #                 callbacks=[thread_safe_printer.print_chunk_callback],
    #                 debug=False,
    #             )
    #             session_service.upsert_session(session)
    #             success_msg = json.dumps({
    #                 "type": "status",
    #                 "data": "completed",
    #                 "timestamp": datetime.utcnow().isoformat()
    #             })
    #             asyncio.run_coroutine_threadsafe(ws_manager.send_message(session.id, success_msg), main_loop)
    #             asyncio.run_coroutine_threadsafe(ws_manager.close_session(session.id), main_loop)
    #             logger.info("Task completed successfully for session %s", session.id)
    #         except Exception as exc:
    #             logger.error("Error in thread job session %s: %s", session.id, exc, exc_info=True)
    #             err_msg = json.dumps({
    #                 "type": "status",
    #                 "data": f"error: {exc}",
    #                 "timestamp": datetime.utcnow().isoformat()
    #             })
    #             asyncio.run_coroutine_threadsafe(ws_manager.send_message(session.id, err_msg), main_loop)
    #             asyncio.run_coroutine_threadsafe(ws_manager.close_session(session.id, code=1011), main_loop)
    #     try:
    #         local_loop.run_until_complete(_async_job())
    #     finally:
    #         local_loop.close()

    def run_agent_sync_wrapper(
        task_request: TaskRequest,
        session: SessionMetadata,
        runner: AgentRunner,
        ws_manager: WebSocketManager,
        session_service: SessionService,
        main_loop: asyncio.AbstractEventLoop,
    ):
        local_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(local_loop)

        workspace = get_session_workspace(session.id)
        tracer = Tracer(file_path=workspace / "tracing.yaml", reset=True)

        # Register session for cancellation tracking
        cancel_event = cancellation_registry.register(session.id, runner)

        # Thread-safe queue for ordered message delivery
        send_queue: queue.Queue[str | None] = queue.Queue()
        sender_task: concurrent.futures.Future[None] | None = None

        async def _sender_loop():
            """Consume messages from queue and send to WebSocket in order."""
            while True:
                try:
                    # Non-blocking check with small sleep to yield control
                    try:
                        msg = send_queue.get_nowait()
                    except queue.Empty:
                        await asyncio.sleep(0.01)
                        continue

                    if msg is None:  # Poison pill signals shutdown
                        break
                    await ws_manager.send_message(session.id, msg)
                except Exception as exc:
                    logger.error("Sender loop error for %s: %s", session.id, exc)

        def _enqueue_message(
            msg_type: str,
            content: str,
            message_id: str | None = None,
            truncated: bool = False,
        ):
            """Thread-safe enqueue of a message (preserves order via FIFO queue)."""
            if not content:
                return
            payload: dict[str, Any] = {
                "type": msg_type,
                "data": str(content),
                "timestamp": datetime.utcnow().isoformat(),
            }
            if message_id:
                payload["message_id"] = message_id
            if truncated:
                payload["truncated"] = True
            msg = json.dumps(payload)
            send_queue.put(msg)

        async def ui_callback(mode: str, chunk: Any):
            # Check for cancellation before processing each chunk
            if cancel_event.is_set():
                raise asyncio.CancelledError("Agent execution cancelled by user")

            if mode == "messages":
                msg = chunk[0] if isinstance(chunk, tuple) else chunk

                if isinstance(msg, AIMessageChunk):
                    msg_id = getattr(msg, "id", None)
                    _enqueue_message("token", str(msg.content), message_id=msg_id)

                elif isinstance(msg, ToolMessage):
                    preview = (
                        str(msg.content)[:500] + "..."
                        if len(str(msg.content)) > 500
                        else str(msg.content)
                    )
                    _enqueue_message("tool", f"Tool Output: {preview}")

                elif isinstance(msg, (HumanMessage, HumanMessageChunk)):
                    full_content = str(msg.content)
                    is_truncated = len(full_content) > 500
                    _enqueue_message(
                        "environment", full_content, truncated=is_truncated
                    )

        async def _async_job():
            nonlocal sender_task
            # Start the sender loop on the main event loop
            sender_task = asyncio.run_coroutine_threadsafe(_sender_loop(), main_loop)

            had_error = False
            was_cancelled = False
            try:
                await runner.astream(
                    task_request.task,
                    callbacks=[tracer.tracing_callback, ui_callback],
                    debug=True,
                )

                session_service.upsert_session(session)
                _enqueue_message("status", "completed")
                logger.info("Task completed successfully for session %s", session.id)

            except asyncio.CancelledError:
                was_cancelled = True
                logger.info("Task cancelled by user for session %s", session.id)
                _enqueue_message("status", "cancelled")

            except Exception as exc:
                had_error = True
                logger.error(
                    "Error in thread job session %s: %s", session.id, exc, exc_info=True
                )
                _enqueue_message("status", f"Error: {str(exc)}")

            finally:
                # Unregister from cancellation registry
                cancellation_registry.unregister(session.id)

                # Signal sender to stop and wait for queue to drain
                send_queue.put(None)
                if sender_task:
                    try:
                        sender_task.result(timeout=10.0)
                    except Exception as exc:
                        logger.warning("Sender task cleanup error: %s", exc)

                # Close WebSocket session after all messages sent
                close_code = 1011 if had_error else 1000
                asyncio.run_coroutine_threadsafe(
                    ws_manager.close_session(session.id, code=close_code), main_loop
                )

        try:
            local_loop.run_until_complete(_async_job())
        finally:
            local_loop.close()

    def build_runner(
        task_request: TaskRequest, session: SessionMetadata
    ) -> AgentRunner:
        logger.info(f"Building agent runner for session: {session.id}")
        merged_opts = _merge_agent_options(task_request)
        workspace = Path(
            merged_opts.get("project_path") or get_session_workspace(session.id)
        )
        merged_opts["project_path"] = str(workspace.resolve())
        app_config = load_config(
            provider=merged_opts.get("provider"),
            model=merged_opts.get("model"),
            model_base_url=merged_opts.get("model_base_url"),
            api_key=merged_opts.get("api_key"),
            max_steps=merged_opts.get("max_steps"),
            config_file=merged_opts.get("config_file"),
        )
        project_path = merged_opts.get("project_path")
        recursion_limit = merged_opts.get("max_steps") or 200
        agent = AutoDSAgent(app_config=app_config, project_path=project_path)
        return AgentRunner(
            agent=agent,
            project_path=project_path,
            recursion_limit=recursion_limit,
            session=session,
        )

    @app.post("/api/sessions", response_model=SessionResponse)
    async def create_session():
        session_service = SessionService()
        session = session_service.create_session()
        return SessionResponse(
            id=session.id, created_at=session.created_at, updated_at=session.updated_at
        )

    @app.get("/api/sessions", response_model=List[SessionResponse])
    async def list_sessions():
        session_service = SessionService()
        return [
            SessionResponse(id=s.id, created_at=s.created_at, updated_at=s.updated_at)
            for s in session_service.list_sessions()
        ]

    @app.get("/api/sessions/{session_id}", response_model=SessionResponse)
    async def get_session(session_id: str):
        session_service = SessionService()
        try:
            s = session_service.get_session(session_id)
            return SessionResponse(
                id=s.id, created_at=s.created_at, updated_at=s.updated_at
            )
        except SessionNotFoundError:
            raise HTTPException(status_code=404)

    @app.post("/api/execute", response_model=TaskResponse)
    async def execute_task(request: TaskRequest):
        session_service = SessionService()
        if request.session_id:
            try:
                session = session_service.get_session(request.session_id)
            except SessionNotFoundError:
                session = session_service.create_session()
        else:
            session = session_service.create_session()
        runner = build_runner(request, session)
        main_loop = asyncio.get_running_loop()
        import threading

        t = threading.Thread(
            target=run_agent_sync_wrapper,
            kwargs={
                "task_request": request,
                "session": session,
                "runner": runner,
                "ws_manager": manager,
                "session_service": session_service,
                "main_loop": main_loop,
            },
            daemon=True,
        )
        t.start()
        return TaskResponse(
            session_id=session.id,
            status="started",
            message="Agent started in background thread",
        )

    @app.post("/api/chat")
    async def chat_message(request: ChatMessage):
        session_service = SessionService()
        if request.session_id:
            try:
                session = session_service.get_session(request.session_id)
            except SessionNotFoundError:
                raise HTTPException(status_code=404)
        else:
            session = session_service.create_session()
        task_request = TaskRequest(task=request.message, session_id=session.id)
        runner = build_runner(task_request, session)
        main_loop = asyncio.get_running_loop()
        import threading

        t = threading.Thread(
            target=run_agent_sync_wrapper,
            kwargs={
                "task_request": task_request,
                "session": session,
                "runner": runner,
                "ws_manager": manager,
                "session_service": session_service,
                "main_loop": main_loop,
            },
            daemon=True,
        )
        t.start()
        return {"status": "started", "session_id": session.id}

    @app.post("/api/session/{session_id}/cancel")
    async def cancel_session(session_id: str):
        """Cancel a running agent session."""
        _ensure_session_exists(session_id)

        if cancellation_registry.cancel(session_id):
            # Send cancelling status via WebSocket
            cancel_msg = json.dumps(
                {
                    "type": "status",
                    "data": "cancelling",
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
            await manager.send_message(session_id, cancel_msg)
            return {"status": "cancelling", "session_id": session_id}
        else:
            return {"status": "not_running", "session_id": session_id}

    @app.post("/api/session/{session_id}/dataset")
    async def upload_dataset(session_id: str, files: List[UploadFile] = File(...)):
        session_service = SessionService()
        try:
            session_service.get_session(session_id)
        except SessionNotFoundError:
            raise HTTPException(status_code=404, detail="Session not found")

        workspace = get_session_workspace(session_id)
        uploaded_paths = []

        for file in files:
            filename = Path(file.filename or "").name
            ext = Path(filename).suffix.lower()
            if ext not in ALLOWED_UPLOAD_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"File type '{ext}' not allowed. Allowed: {', '.join(sorted(ALLOWED_UPLOAD_EXTENSIONS))}",
                )

            dest = workspace / filename
            content = await file.read()
            dest.write_bytes(content)
            uploaded_paths.append(filename)
            logger.info("Dataset %s uploaded for session %s", filename, session_id)

        return {"paths": uploaded_paths}

    @app.websocket("/api/ws/{session_id}")
    async def websocket_endpoint(websocket: WebSocket, session_id: str):
        await manager.connect(websocket, session_id)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            await manager.disconnect(session_id, websocket)
        except Exception:
            await manager.disconnect(session_id, websocket)

    @app.get("/api/session/{session_id}/history")
    async def get_session_history(session_id: str):
        session_service = SessionService()
        try:
            session = session_service.get_session(session_id)
            task_request = TaskRequest(task="", session_id=session_id)
            runner = build_runner(task_request, session)
            state = await runner.get_state()
            messages = state.values.get("messages") or [] if state else []
            return {
                "session_id": session.id,
                "messages": [
                    {"type": type(msg).__name__, "content": msg.content}
                    for msg in messages
                ],
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/config")
    async def get_config():
        from autods.constants import DEFAULT_CONFIG_PATH

        return {
            "yaml": DEFAULT_CONFIG_PATH.read_text()
            if DEFAULT_CONFIG_PATH.exists()
            else ""
        }

    @app.post("/api/config")
    async def update_config(request: Dict[str, str]):
        from autods.constants import DEFAULT_CONFIG_PATH

        try:
            yaml.safe_load(request.get("yaml", ""))
            if not DEFAULT_CONFIG_PATH.parent.exists():
                DEFAULT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            DEFAULT_CONFIG_PATH.write_text(request.get("yaml", ""))
            return {"message": "Success"}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}

    # Dataset management endpoints (grad.py integration)
    @app.get("/api/datasets")
    async def list_datasets():
        """List all indexed datasets from grad."""
        try:
            from autods.grad.grad import grad

            datasets = await grad.list_datasets()
            return {
                "datasets": [
                    {"name": d.name, "id": str(d.id)} for d in (datasets or [])
                ]
            }
        except Exception as e:
            logger.error("Failed to list datasets: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/datasets")
    async def add_dataset(request: AddDatasetRequest):
        """Add a repository to the knowledge graph (grad.add)."""
        try:
            from autods.grad.grad import grad
            from autods.grad.repository import get_repository_id

            await grad.add(request.url)
            repo_id = get_repository_id(request.url)
            return {"status": "success", "name": repo_id, "url": request.url}
        except Exception as e:
            logger.error("Failed to add dataset: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/datasets/{name}")
    async def delete_dataset(name: str):
        """Delete a dataset from the knowledge graph."""
        try:
            from autods.grad.grad import grad

            await grad.delete_dataset(name)
            return {"status": "deleted", "name": name}
        except Exception as e:
            logger.error("Failed to delete dataset: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    # Library installation endpoint
    @app.post("/api/session/{session_id}/install")
    async def install_libraries(session_id: str, request: InstallLibrariesRequest):
        """Install Python libraries in the session's venv."""
        _ensure_session_exists(session_id)
        workspace = get_session_workspace(session_id)

        if not request.libraries:
            return {"status": "no_libraries", "message": "No libraries specified"}

        # Create venv if it doesn't exist
        venv_path = workspace / ".venv"
        pip_path = venv_path / "bin" / "pip"

        import subprocess

        try:
            # Create venv if needed
            if not venv_path.exists():
                subprocess.run(
                    ["python3", "-m", "venv", str(venv_path)],
                    check=True,
                    capture_output=True,
                )
                logger.info("Created venv at %s", venv_path)

            # Install libraries
            result = subprocess.run(
                [str(pip_path), "install"] + request.libraries,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            if result.returncode != 0:
                raise HTTPException(
                    status_code=500,
                    detail=f"pip install failed: {result.stderr}",
                )

            return {
                "status": "success",
                "installed": request.libraries,
                "output": result.stdout[-1000:] if result.stdout else "",
            }
        except subprocess.TimeoutExpired:
            raise HTTPException(
                status_code=500, detail="Installation timed out after 5 minutes"
            )
        except Exception as e:
            logger.error("Failed to install libraries: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    def _ensure_session_exists(session_id: str):
        try:
            SessionService().get_session(session_id)
        except SessionNotFoundError:
            raise HTTPException(status_code=404, detail="Session not found")

    @app.post("/api/session/{session_id}/artifacts/snapshot")
    async def refresh_snapshot(session_id: str):
        _ensure_session_exists(session_id)
        return {"status": "ok"}

    def _get_artifacts_sync(session_id: str):
        root = _artifacts_root(session_id)
        if not root.exists():
            return {"root": str(root), "tree": [], "files": [], "hash": ""}
        count = 0
        ARTIFACT_TREE_MAX_ITEMS = 10000

        def scan_dir(path, current_depth=0):
            nonlocal count
            items = []
            if current_depth > ARTIFACT_TREE_MAX_DEPTH:
                return []
            try:
                entries = sorted(
                    path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())
                )
            except PermissionError:
                return []
            for entry in entries:
                if count > ARTIFACT_TREE_MAX_ITEMS:
                    break
                if entry.name == "__pycache__":
                    continue
                rel_path = entry.relative_to(root).as_posix()
                if entry.is_dir():
                    children = scan_dir(entry, current_depth + 1)
                    items.append(
                        {
                            "type": "directory",
                            "name": entry.name,
                            "path": rel_path,
                            "children": children,
                        }
                    )
                else:
                    count += 1
                    items.append(
                        {
                            "type": "file",
                            "name": entry.name,
                            "path": rel_path,
                            "size": entry.stat().st_size,
                        }
                    )
            return items

        tree = scan_dir(root)
        tree_str = json.dumps(tree, sort_keys=True, default=str)
        data_hash = hashlib.md5(tree_str.encode("utf-8")).hexdigest()
        return {"root": str(root), "tree": tree, "files": [], "hash": data_hash}

    def _create_zip_sync(root: Path) -> io.BytesIO:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for obj in root.rglob("*"):
                if obj.is_dir():
                    continue
                try:
                    zf.write(obj, obj.relative_to(root).as_posix())
                except Exception as e:
                    logger.warning(f"Could not zip file {obj}: {e}")
        buffer.seek(0)
        return buffer

    @app.get("/api/session/{session_id}/artifacts")
    async def get_artifacts(session_id: str):
        _ensure_session_exists(session_id)
        return await run_in_threadpool(_get_artifacts_sync, session_id)

    @app.get("/api/session/{session_id}/file")
    async def get_file(session_id: str, file_path: str):
        _ensure_session_exists(session_id)
        root = _artifacts_root(session_id).resolve()
        requested = (root / file_path).resolve()
        if not str(requested).startswith(str(root)):
            raise HTTPException(status_code=400)
        if not requested.exists():
            raise HTTPException(status_code=404)
        return FileResponse(requested)

    @app.get("/api/session/{session_id}/artifacts/archive")
    async def download_artifact_archive(session_id: str):
        _ensure_session_exists(session_id)
        root = _artifacts_root(session_id).resolve()
        if not root.exists():
            raise HTTPException(status_code=404, detail="No artifacts")
        buffer = await run_in_threadpool(_create_zip_sync, root)
        filename = f"{session_id}_artifacts.zip"
        return StreamingResponse(
            buffer,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    return app
