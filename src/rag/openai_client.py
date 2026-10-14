import os
import openai
from config import config

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
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        self.embed_model = embed_model or config.openai_embed_model or "text-embedding-ada-002"
        self.is_configured = bool(self.api_key)

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response from the chat model.
        Returns the full response text.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return response.choices[0].message.content.strip()

    def generate_stream(self, prompt: str, **kwargs):
        """Yield tokens from a streaming chat completion for a simple prompt.
        This method is retained for backward compatibility.
        """
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            **kwargs,
        )
        for chunk in stream:
            if hasattr(chunk.choices[0].delta, "content"):
                yield chunk.choices[0].delta.content

    def chat_stream(self, messages: list[dict[str, str]], system: str | None = None, **kwargs):
        """Yield tokens from a streaming chat completion given a list of messages.
        If a system prompt is provided, it is inserted as the first message with role 'system'.
        """
        chat_messages = []
        if system:
            chat_messages.append({"role": "system", "content": system})
        chat_messages.extend(messages)
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=chat_messages,
            stream=True,
            **kwargs,
        )
        for chunk in stream:
            if hasattr(chunk.choices[0].delta, "content"):
                yield chunk.choices[0].delta.content

    def embed(self, texts: list[str]) -> list[float]:
        """Return embedding vector for a list of texts (concatenated).
        For simplicity, returns embedding of the first text.
        """
        response = self.client.embeddings.create(model=self.embed_model, input=texts)
        # Return list of floats for the first input
        return response.data[0].embedding
