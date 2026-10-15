import os
import openai
from config import config

from src.security.logger import get_secure_logger

logger = get_secure_logger("openai_client")

def is_context_length_exceeded(err: Exception) -> bool:
    """Detects OpenAI context length exceeded errors across SDK formats and exceptions."""
    s = f"{err} {repr(err)}".lower()
    if any(k in s for k in ("context_length_exceeded", "maximum context length", "reduce the length of the messages")):
        return True
    body = getattr(err, "body", None)
    if isinstance(body, dict):
        err_info = body.get("error")
        if isinstance(err_info, dict) and err_info.get("code") == "context_length_exceeded":
            return True
    return False


class OpenAIClient:
    """Simple OpenAI client wrapper supporting chat completion and embeddings.
    Uses the API key from configuration or supplied explicitly.
    """

    def __init__(self, api_key: str | None = None, model: str | None = None, embed_model: str | None = None):
        """Initialize the OpenAI client.
        The client now tolerates a missing API key. Instead of raising an exception during construction,
        we store the key (or ``None``) and defer validation until a request is made. This allows
        ``ConversationalRAGEngine`` to emit a friendly Korean error message without falling back to another provider.
        """
        # Use provided api_key if given (including empty string), otherwise fall back to config
        if api_key is not None:
            self.api_key = api_key
        else:
            self.api_key = config.openai_api_key
        # Initialise OpenAI client (v1 SDK)
        # If api_key is empty, the client will still be created but requests will fail; we handle that later.
        self.client = openai.OpenAI(api_key=self.api_key) if self.api_key else openai.OpenAI()
        self.model = model or getattr(config, "openai_model", None) or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.embed_model = embed_model or config.openai_embed_model or "text-embedding-ada-002"
        self.is_configured = bool(self.api_key)

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response from the chat model.
        Returns the full response text.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if is_context_length_exceeded(e) and "3.5" in self.model:
                logger.info(f"Context length exceeded for {self.model}. Auto-upgrading to gpt-4o-mini (128k context).")
                self.model = "gpt-4o-mini"
                return self.generate(prompt, **kwargs)
            raise

    def generate_stream(self, prompt: str, **kwargs):
        """Yield tokens from a streaming chat completion for a simple prompt.
        This method is retained for backward compatibility.
        """
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                **kwargs,
            )
        except Exception as e:
            if is_context_length_exceeded(e) and "3.5" in self.model:
                logger.info(f"Context length exceeded for {self.model}. Auto-upgrading to gpt-4o-mini (128k context).")
                self.model = "gpt-4o-mini"
                yield from self.generate_stream(prompt, **kwargs)
                return
            raise

        for chunk in stream:
            if hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def chat_stream(self, messages: list[dict[str, str]], system: str | None = None, **kwargs):
        """Yield tokens from a streaming chat completion given a list of messages.
        If a system prompt is provided, it is inserted as the first message with role 'system'.
        Includes auto-recovery for context_length_exceeded.
        """
        chat_messages = []
        if system:
            chat_messages.append({"role": "system", "content": system})
        chat_messages.extend(messages)

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=chat_messages,
                stream=True,
                **kwargs,
            )
        except Exception as e:
            if is_context_length_exceeded(e):
                logger.warning(f"OpenAI context_length_exceeded on model {self.model}: {e}")
                # 1. Auto-upgrade from gpt-3.5-turbo (16k) to gpt-4o-mini (128k context)
                if "3.5" in self.model:
                    logger.info("Automatically upgrading model from gpt-3.5-turbo to gpt-4o-mini (128k window).")
                    self.model = "gpt-4o-mini"
                    yield from self.chat_stream(messages, system=system, **kwargs)
                    return
                # 2. Prune older chat history turns to recover context length
                if len(chat_messages) > 2:
                    logger.info("Pruning older chat history turns to recover context length...")
                    condensed = [chat_messages[0], chat_messages[-1]]  # Keep system + latest user message
                    stream = self.client.chat.completions.create(
                        model=self.model,
                        messages=condensed,
                        stream=True,
                        **kwargs,
                    )
                else:
                    raise
            else:
                raise

        for chunk in stream:
            if hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def embed(self, texts: list[str]) -> list[float]:
        """Return embedding vector for a list of texts (concatenated).
        For simplicity, returns embedding of the first text.
        """
        response = self.client.embeddings.create(model=self.embed_model, input=texts)
        # Return list of floats for the first input
        return response.data[0].embedding
