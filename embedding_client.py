"""LMStudio embedding client using the OpenAI-compatible API."""

import sys
from openai import OpenAI, APIConnectionError, APITimeoutError, APIStatusError


class LMStudioClient:
    """Client for LMStudio's OpenAI-compatible embedding endpoint."""

    def __init__(self, base_url: str = "http://localhost:1234/v1", model: str = "qwen-0.6b"):
        self.base_url = base_url
        self.model = model
        self.client = OpenAI(
            base_url=base_url,
            api_key="lm-studio",  # LMStudio does not require a real key
            timeout=30.0,
        )

    def _prepare_text(self, text: str) -> str:
        """Replace newlines with spaces -- standard practice for embeddings."""
        return text.replace("\n", " ")

    def get_embedding(self, text: str) -> list[float]:
        """Embed a single text string and return its embedding vector."""
        cleaned = self._prepare_text(text)
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=cleaned,
            )
            return response.data[0].embedding
        except APIConnectionError:
            print(
                f"Error: Cannot connect to LMStudio at {self.base_url}. "
                "Make sure LMStudio is running and the embedding model is loaded.",
                file=sys.stderr,
            )
            raise
        except APITimeoutError:
            print(
                f"Error: Request to LMStudio timed out after 30s. "
                "The server may be overloaded.",
                file=sys.stderr,
            )
            raise
        except APIStatusError as e:
            print(
                f"Error: LMStudio returned status {e.status_code}: {e.message}",
                file=sys.stderr,
            )
            raise

    def get_embeddings(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Embed multiple texts, chunking into batches to avoid API limits."""
        if not texts:
            return []

        cleaned = [self._prepare_text(t) for t in texts]
        all_embeddings: list[list[float]] = []

        for i in range(0, len(cleaned), batch_size):
            chunk = cleaned[i : i + batch_size]
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=chunk,
                )
                all_embeddings.extend(item.embedding for item in response.data)
            except APIConnectionError:
                print(
                    f"Error: Cannot connect to LMStudio at {self.base_url}. "
                    "Make sure LMStudio is running and the embedding model is loaded.",
                    file=sys.stderr,
                )
                raise
            except APITimeoutError:
                print(
                    f"Error: Request to LMStudio timed out after 30s. "
                    "The server may be overloaded.",
                    file=sys.stderr,
                )
                raise
            except APIStatusError as e:
                print(
                    f"Error: LMStudio returned status {e.status_code}: {e.message}",
                    file=sys.stderr,
                )
                raise

        return all_embeddings
