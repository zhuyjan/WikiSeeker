import time

import requests

DEFAULT_RETRIEVER_SERVICE_URL = "http://localhost:5678"


class RetrieverClient:
    """Client for communicating with the standalone retriever service."""

    def __init__(self, service_url=DEFAULT_RETRIEVER_SERVICE_URL, max_retries=10):
        self.service_url = service_url
        self.max_retries = max_retries
        self._wait_for_service()

    def _wait_for_service(self):
        """Wait for the retriever service to be ready."""
        print(f"[INFO] Waiting for retriever service at {self.service_url}...")
        for i in range(self.max_retries):
            try:
                response = requests.get(f"{self.service_url}/health", timeout=5)
                if response.status_code == 200:
                    print("[INFO] Retriever service is ready")
                    return
            except requests.exceptions.RequestException:
                pass

            if i < self.max_retries - 1:
                print(
                    "[INFO] Service not ready, retrying in 3 seconds... "
                    f"({i + 1}/{self.max_retries})"
                )
                time.sleep(3)

        raise RuntimeError(
            f"Failed to connect to retriever service at {self.service_url}. "
            "Please make sure the service is running by executing: "
            "bash scripts/start_retriever_service.sh"
        )

    def search_by_path(self, img_path, query, top_k=200, alpha=0.6):
        """
        Perform search using the retriever service with image path.

        Args:
            img_path: Path to the image file
            query: Search query string
            top_k: Number of top results to return
            alpha: Weight for visual embedding (0-1), text weight is (1-alpha)

        Returns:
            List of search results
        """
        try:
            response = requests.post(
                f"{self.service_url}/search",
                json={
                    "query": query,
                    "img_path": img_path,
                    "top_k": top_k,
                    "alpha": alpha,
                },
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()
            if result.get("success"):
                return result["results"]
            raise RuntimeError(f"Search failed: {result.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"[ERROR] Failed to communicate with retriever service: {e}")
            raise


_retriever_client = None


def get_retriever(service_url=DEFAULT_RETRIEVER_SERVICE_URL, max_retries=10):
    """Get or initialize the global retriever client instance."""
    global _retriever_client
    if (
        _retriever_client is None
        or _retriever_client.service_url != service_url
        or _retriever_client.max_retries != max_retries
    ):
        print("[INFO] Initializing RetrieverClient...")
        _retriever_client = RetrieverClient(
            service_url=service_url,
            max_retries=max_retries,
        )
        print("[INFO] RetrieverClient initialized successfully")
    return _retriever_client
