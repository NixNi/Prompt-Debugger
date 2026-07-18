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

    def get_embeddings(self, texts: list[str], batch_size: int = 32, runs: int = 1) -> list[list[float]]:
        """Embed multiple texts, chunking into batches. Optionally run N times and average."""
        if not texts:
            return []

        import numpy as np

        all_runs = []
        for run in range(runs):
            if runs > 1:
                print(f"  Run {run + 1}/{runs}...")
            cleaned = [self._prepare_text(t) for t in texts]
            run_embeddings: list[list[float]] = []

            for i in range(0, len(cleaned), batch_size):
                chunk = cleaned[i : i + batch_size]
                try:
                    response = self.client.embeddings.create(
                        model=self.model,
                        input=chunk,
                    )
                    run_embeddings.extend(item.embedding for item in response.data)
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

            all_runs.append(run_embeddings)

        if len(all_runs) == 1:
            return all_runs[0]

        averaged = np.mean(all_runs, axis=0).tolist()
        return averaged
