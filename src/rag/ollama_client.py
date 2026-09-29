"""
Ollama Local API Client.
Communicates with locally running Ollama instance (default: http://localhost:11434).
Supports listing installed models, generating streaming chat/completions, and generating embeddings.
"""
import json
from typing import List, Dict, Any, Generator, Optional
import requests
from src.security.logger import get_secure_logger

logger = get_secure_logger("ollama_client")


class OllamaClient:
    """Lightweight HTTP client for local Ollama server."""

    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def check_connection(self) -> Dict[str, Any]:
        """
        Checks if Ollama server is reachable and lists installed models.
        Returns: {"connected": bool, "models": list, "error": str|None}
        """
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("name", "") for m in data.get("models", [])]
                return {"connected": True, "models": models, "error": None}
            return {"connected": False, "models": [], "error": f"HTTP {resp.status_code}"}
        except Exception as e:
            return {"connected": False, "models": [], "error": str(e)}

    def embed(self, model: str, texts: List[str]) -> List[List[float]]:
        """
        Generates vector embeddings using Ollama.
        Supports both modern /api/embed and legacy /api/embeddings.
        """
        if not texts:
            return []

        # Try modern /api/embed (batch support)
        try:
            resp = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": model, "input": texts},
                timeout=self.timeout
            )
            if resp.status_code == 200:
                data = resp.json()
                if "embeddings" in data:
                    return data["embeddings"]
        except Exception as e:
            logger.debug(f"/api/embed not available or failed: {e}. Trying /api/embeddings endpoint.")

        # Fallback to single /api/embeddings loop
        results = []
        for text in texts:
            resp = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": model, "prompt": text},
                timeout=self.timeout
            )
            if resp.status_code == 200:
                results.append(resp.json().get("embedding", []))
            else:
                raise RuntimeError(f"Ollama embedding error (HTTP {resp.status_code}): {resp.text}")
        return results

    def chat_stream(
        self,
        model: str,
        messages: List[Dict[str, str]],
        system: Optional[str] = None
    ) -> Generator[str, None, None]:
        """
        Streams chat completion tokens from Ollama /api/chat.
        """
        payload_messages = []
        if system:
            payload_messages.append({"role": "system", "content": system})
        payload_messages.extend(messages)

        payload = {
            "model": model,
            "messages": payload_messages,
            "stream": True,
            "options": {
                "temperature": 0.3,
                "top_p": 0.9
            }
        }

        with requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            stream=True,
            timeout=self.timeout
        ) as resp:
            if resp.status_code != 200:
                raise RuntimeError(f"Ollama chat error (HTTP {resp.status_code}): {resp.text}")

            for line in resp.iter_lines(decode_unicode=True):
                if line:
                    try:
                        chunk = json.loads(line)
                        msg = chunk.get("message", {})
                        content = msg.get("content", "")
                        if content:
                            yield content
                        if chunk.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue

    def generate(self, model: str, prompt: str) -> str:
        """
        Non-streaming generation from Ollama /api/generate (useful for query rewriting).
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2}
        }
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout
        )
        if resp.status_code == 200:
            return resp.json().get("response", "").strip()
        raise RuntimeError(f"Ollama generate error: {resp.status_code}")
