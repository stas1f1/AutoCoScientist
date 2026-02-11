import httpx


class GradClient:
    """HTTP client for the remote pygrad REST server."""

    def __init__(self, host: str, port: int, timeout: float = 300.0):
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout

    async def add(self, url: str) -> dict:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{self.base_url}/repos", json={"url": url})
            resp.raise_for_status()
            return resp.json()

    async def list_repos(self) -> list[dict]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(f"{self.base_url}/repos")
            resp.raise_for_status()
            return resp.json()

    async def search(self, url: str, query: str) -> str:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/repos/search",
                json={"url": url, "query": query},
            )
            resp.raise_for_status()
            return resp.json()["result"]

    async def delete(self, url: str) -> dict:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.delete(f"{self.base_url}/repos", params={"url": url})
            resp.raise_for_status()
            return resp.json()

    async def visualize(self) -> str:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(f"{self.base_url}/visualize")
            resp.raise_for_status()
            return resp.text
