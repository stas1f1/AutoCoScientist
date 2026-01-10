import json
import os
import queue
import re
import threading
import time
from typing import Dict, List, Optional

import requests
import streamlit as st
import streamlit.components.v1 as components
import websocket
from app_components.multi_file_uploader import render_multi_uploader
from previewer import ArtifactPreviewer
from streamlit import fragment

st.set_page_config(
    page_title="AutoDS Agent Interface", layout="wide", initial_sidebar_state="expanded"
)
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
WS_BASE_URL = API_BASE_URL.replace("http", "ws")
artifact_previewer = ArtifactPreviewer(API_BASE_URL)


def init_session_state():
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = None
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "artifact_preview" not in st.session_state:
        st.session_state.artifact_preview = None
    if "artifact_zip" not in st.session_state:
        st.session_state.artifact_zip = None
    if "artifact_zip_name" not in st.session_state:
        st.session_state.artifact_zip_name = ""
    if "ws_connections" not in st.session_state:
        st.session_state.ws_connections = {}
    if "session_logs" not in st.session_state:
        st.session_state.session_logs = {}
    if "artifact_data" not in st.session_state:
        st.session_state.artifact_data = {}
    if "just_created_session" not in st.session_state:
        st.session_state.just_created_session = None
    if "yaml_config" not in st.session_state:
        st.session_state.yaml_config = ""
    if "artifact_hash" not in st.session_state:
        st.session_state.artifact_hash = {}


def upload_dataset(session_id: str, uploaded_files: list) -> bool:
    if not uploaded_files:
        return False

    files_payload = [
        ("files", (file.name, file.getvalue(), "text/csv")) for file in uploaded_files
    ]

    try:
        res = requests.post(
            f"{API_BASE_URL}/api/session/{session_id}/dataset",
            files=files_payload,
            timeout=120,
        )

        if not res.ok:
            st.error(f"Dataset upload failed: {res.text}")
            return False

        return True
    except Exception as e:
        st.error(f"Upload error: {e}")
        return False


def check_api_connection() -> bool:
    try:
        return requests.get(f"{API_BASE_URL}/health", timeout=5).status_code == 200
    except Exception:
        return False


def create_session() -> Optional[Dict]:
    try:
        return requests.post(f"{API_BASE_URL}/api/sessions").json()
    except Exception:
        return None


def get_sessions() -> List[Dict]:
    try:
        return requests.get(f"{API_BASE_URL}/api/sessions").json()
    except Exception:
        return []


def fetch_artifacts(session_id: str) -> Dict:
    try:
        t = int(time.time())
        url = f"{API_BASE_URL}/api/session/{session_id}/artifacts?t={t}"
        return requests.get(url, timeout=5).json()
    except Exception:
        return {"tree": [], "files": []}


def start_ws_listener(session_id: str):
    if session_id in st.session_state.ws_connections:
        return
    event_queue: queue.Queue = queue.Queue()
    stop_event = threading.Event()

    def on_message(ws, message):
        event_queue.put(message)

    def on_close(ws, status_code, msg):
        if not stop_event.is_set():
            time.sleep(1)
            try:
                start_ws_listener(session_id)
            except Exception:
                pass

    ws_url = f"{WS_BASE_URL}/api/ws/{session_id}"
    ws_app = websocket.WebSocketApp(
        ws_url,
        on_message=on_message,
        on_close=on_close,
    )
    thread = threading.Thread(
        target=lambda: ws_app.run_forever(ping_interval=30, ping_timeout=5),
        daemon=True,
    )
    thread.start()
    st.session_state.ws_connections[session_id] = {
        "thread": thread,
        "queue": event_queue,
        "stop": stop_event,
    }


def process_ws_events():
    for session_id, conn in list(st.session_state.ws_connections.items()):
        q: queue.Queue = conn["queue"]
        while True:
            try:
                raw = q.get_nowait()
            except queue.Empty:
                break
            try:
                payload = json.loads(raw)
                logs = st.session_state.session_logs.setdefault(session_id, [])
                logs.append(payload)
                if len(logs) > 50000:
                    del logs[:-50000]
            except Exception:
                continue


@fragment(run_every=1.0)
def render_logs_fragment(session_id):
    if not session_id:
        st.info("Сессия не выбрана")
        return

    process_ws_events()
    logs = st.session_state.session_logs.get(session_id, [])

    if not logs:
        st.info("Ожидание логов от агента...")
        return

    full_buffer = ""
    for entry in logs:
        etype = entry.get("type")
        data = entry.get("data", "")

        if etype == "token":
            full_buffer += data
        elif etype == "tool":
            clean_data = data.replace("```", "")
            full_buffer += f"\n\n<details><summary>🛠 <b>Tool Output</b></summary>\n\n```text\n{clean_data}\n```\n\n</details>\n\n"
        elif etype == "status":
            full_buffer += f"\n\n*ℹ️ {data}*\n\n"
        elif etype == "info":
            full_buffer += f"\n`{data}`\n"

    full_buffer = re.sub(
        r'<CodeBlock\s+lang="([^"]+)">\s*(`{3,}\w*)?', r"\n```\1\n", full_buffer
    )
    full_buffer = re.sub(r"(`{3,})?\s*</CodeBlock>", r"\n```\n", full_buffer)

    anchor_id = f"log_end_{session_id}"

    with st.container(height=600, border=True):
        st.markdown(full_buffer, unsafe_allow_html=True)
        st.markdown(f'<div id="{anchor_id}"></div>', unsafe_allow_html=True)

    js = f"""
    <script>
        var element = window.parent.document.getElementById("{anchor_id}");
        if (element) {{
            element.scrollIntoView({{behavior: "smooth", block: "end", inline: "nearest"}});
        }}
    </script>
    """
    components.html(js, height=0, width=0)


@fragment(run_every=5)
def polling_artifacts(session_id):
    if not session_id:
        return
    try:
        new_data = fetch_artifacts(session_id)
        new_hash = new_data.get("hash", "")
        current_hash = st.session_state.artifact_hash.get(session_id, "")
        if new_hash and new_hash != current_hash:
            st.session_state.artifact_data[session_id] = new_data
            st.session_state.artifact_hash[session_id] = new_hash
            st.rerun()
    except Exception:
        pass


def render_artifacts_ui(session_id):
    if not session_id:
        return
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("🔄 Force Refresh"):
            st.session_state.artifact_hash.pop(session_id, None)
            st.rerun()
    data = st.session_state.artifact_data.get(session_id)
    if not data:
        st.info("Waiting for data...")
        polling_artifacts(session_id)
        return
    tree = data.get("tree", [])
    if not tree:
        st.info("No artifacts found.")
        return

    def draw_tree(nodes):
        MAX_ITEMS = 50
        folders = sorted(
            [n for n in nodes if n.get("type") == "directory"],
            key=lambda x: x["name"].lower(),
        )
        files = sorted(
            [n for n in nodes if n.get("type") != "directory"],
            key=lambda x: x["name"].lower(),
        )
        for i, node in enumerate(folders):
            if i >= MAX_ITEMS:
                st.caption(f"... {len(folders) - i} more folders")
                break
            with st.expander(f"📁 {node['name']}", expanded=False):
                if node.get("children"):
                    draw_tree(node["children"])
                else:
                    st.caption("(empty)")
        for i, node in enumerate(files):
            if i >= MAX_ITEMS:
                st.caption(f"... {len(files) - i} more files")
                break
            c1, c2 = st.columns([0.85, 0.15])
            kb = node.get("size", 0) / 1024
            sz = f"{kb:.1f} KB" if kb > 1 else f"{node.get('size', 0)} B"
            c1.markdown(
                f"<div style='margin-left:5px'>📄 {node['name']} <span style='color:grey;font-size:0.8em'>({sz})</span></div>",
                unsafe_allow_html=True,
            )
            key_path = node.get("path", node["name"])
            if c2.button("Preview", key=f"v_{session_id}_{key_path}"):
                try:
                    prev = artifact_previewer.preview(session_id=session_id, node=node)
                    st.session_state.artifact_preview = {
                        "name": node["name"],
                        "preview": prev,
                    }
                    st.rerun()
                except Exception:
                    pass

    draw_tree(tree)
    st.divider()
    if st.button("📦 Download Full Archive (.zip)"):
        try:
            r = requests.get(
                f"{API_BASE_URL}/api/session/{session_id}/artifacts/archive",
                stream=True,
            )
            if r.ok:
                st.download_button(
                    "⬇️ Save ZIP",
                    r.content,
                    f"{session_id}.zip",
                    "application/zip",
                    key=f"dl_{time.time()}",
                )
        except Exception as e:
            st.error(f"Error: {e}")


def ensure_ws_listener(session_id: Optional[str]):
    if session_id:
        start_ws_listener(session_id)


def main():
    init_session_state()
    api_online = check_api_connection()
    if not api_online:
        st.warning(f"API is not available. Interface is active, updates are automatic.")
    with st.sidebar:
        st.title("AutoDS Control")
        if st.button("New Session"):
            session = create_session()
            if session:
                st.session_state.current_session_id = session["id"]
                st.session_state.chat_messages = []
                st.session_state.just_created_session = session
                st.session_state.artifact_preview = None
                st.session_state.artifact_zip = None
                ensure_ws_listener(session["id"])
                st.rerun()
        sessions = get_sessions()
        just_created = st.session_state.get("just_created_session")
        if just_created and all(s["id"] != just_created["id"] for s in sessions):
            sessions.insert(0, just_created)
        if sessions:
            session_map = {
                f"{s['id'][:8]} ({s['created_at'][11:16]})": s["id"] for s in sessions
            }
            selected = st.selectbox(
                "📂 Select Session", options=list(session_map.keys())
            )
            selected_id = session_map.get(selected)
            if selected_id and selected_id != st.session_state.current_session_id:
                st.session_state.current_session_id = selected_id
                st.session_state.artifact_preview = None
                st.session_state.artifact_zip = None
        ensure_ws_listener(st.session_state.current_session_id)
    tab1, tab2, tab3, tab4 = st.tabs(
        ["💬 Chat", "🚀 Tasks", "📂 Artifacts", "🔧 Configuration"]
    )
    with tab1:
        st.subheader("System Logs (Live)")
        if st.session_state.current_session_id:
            render_logs_fragment(st.session_state.current_session_id)
        st.divider()
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        if prompt := st.chat_input():
            session_id = st.session_state.current_session_id
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            try:
                res = requests.post(
                    f"{API_BASE_URL}/api/chat",
                    json={"message": prompt, "session_id": session_id},
                )
                if res.ok:
                    sid = res.json().get("session_id")
                    st.session_state.current_session_id = sid
                    ensure_ws_listener(sid)
                else:
                    st.session_state.chat_messages.append(
                        {"role": "assistant", "content": f"Error: {res.text}"}
                    )
            except Exception as exc:
                st.session_state.chat_messages.append(
                    {"role": "assistant", "content": f"Connection Error: {exc}"}
                )
            st.rerun()
        # st.divider()
    with tab2:
        st.subheader("One-off Task Execution")
        col1, col2 = st.columns([2, 1])
        with col1:
            task_input_mode = st.radio(
                "Input source",
                options=["Upload .md/.txt file", "Enter text manually"],
                index=0,
                horizontal=True,
            )
            task_text = ""
            if task_input_mode == "Upload .md/.txt file":
                uploaded = st.file_uploader(
                    "Upload task description", type=["md", "txt"]
                )
                if uploaded:
                    task_text = uploaded.read().decode("utf-8").strip()
                    st.code(task_text or "[empty file]", language="markdown")
            else:
                task_text = st.text_area("Describe the task", height=250)
            # dataset = st.file_uploader("Optional dataset (.csv)", type=["csv"])
            attachments = render_multi_uploader(
                label="Загрузите один или несколько файлов задачи",
                key="task_attachments",
                types=["md", "txt", "csv", "zip"],
                help_text="Можно перетащить сразу несколько файлов.",
            )
        with col2:
            provider = st.selectbox(
                "Provider",
                ["", "google", "openai", "anthropic"],
                format_func=lambda x: x or "Use config.yaml",
            )
            model = st.text_input("Model", "")
            api_key = st.text_input("API Key", type="password")
            max_steps = st.number_input("Max Steps", 1, 1000, 100)
        if st.button(
            "Launch Task",
            type="primary",
            use_container_width=True,
            disabled=not task_text,
        ):
            with st.status("Preparing session..."):
                session = create_session()
                if not session:
                    st.error("Failed to create session")
                    st.stop()
            session_id = session["id"]
            if attachments and not upload_dataset(session_id, attachments):
                st.stop()
            payload = {
                "task": task_text,
                "session_id": session_id,
                "max_steps": max_steps,
            }
            if provider:
                payload["provider"] = provider
            if model:
                payload["model"] = model
            if api_key:
                payload["api_key"] = api_key
            try:
                res = requests.post(f"{API_BASE_URL}/api/execute", json=payload)
                if res.status_code == 200:
                    st.session_state.current_session_id = session_id
                    ensure_ws_listener(session_id)
                    st.success("Started!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"Failed: {res.text}")
            except Exception as exc:
                st.error(f"Error: {exc}")
    with tab3:
        session_id = st.session_state.current_session_id
        if session_id:
            polling_artifacts(session_id)
            render_artifacts_ui(session_id)
            if st.session_state.artifact_preview:
                st.divider()
                name = st.session_state.artifact_preview["name"]
                preview = st.session_state.artifact_preview["preview"]
                if st.button("✖ Close Preview"):
                    st.session_state.artifact_preview = None
                    st.rerun()
                st.markdown(f"### Preview: {name}")
                match preview["type"]:
                    case "text":
                        st.code(preview["text"])
                    case "table":
                        st.dataframe(preview["dataframe"], use_container_width=True)
                    case "image":
                        st.image(
                            preview["image"],
                            caption=f"{preview['format']} {preview['size']}",
                        )
                    case "notebook":
                        components.html(preview["html"], height=800, scrolling=True)
                    case _:
                        st.error(f"Preview error: {preview.get('error')}")
        else:
            st.info("Select a session to view artifacts.")
    with tab4:
        if st.button("Fetch Current Config"):
            try:
                res = requests.get(f"{API_BASE_URL}/api/config").json()
                st.session_state.yaml_config = res.get("yaml", "")
            except Exception as exc:
                st.error(f"Failed to fetch: {exc}")
        if st.session_state.yaml_config:
            updated_yaml = st.text_area(
                "Edit config.yaml", value=st.session_state.yaml_config, height=500
            )
            if st.button("Save & Apply Configuration"):
                try:
                    requests.post(
                        f"{API_BASE_URL}/api/config", json={"yaml": updated_yaml}
                    )
                    st.success("Config saved.")
                    st.session_state.yaml_config = updated_yaml
                except Exception as exc:
                    st.error(f"Failed: {exc}")


if __name__ == "__main__":
    main()
