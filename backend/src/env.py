from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # webrtc TURN servers
    CF_TURN_TOKEN_ID: str
    CF_TURN_API_TOKEN: str

    # STT
    DEEPGRAM_API_KEY: str
    FIREWORKS_API_KEY: str
    # SONIOX_API_KEY: str

    # LLM
    CEREBRAS_BASE_URL: str = "https://api.cerebras.ai/v1"
    CEREBRAS_API_KEY: str

    # OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    # OPENAI_API_KEY: str

    # ANTHROPIC_BASE_URL: str = "https://api.anthropic.com/v1"
    # ANTHROPIC_API_KEY: str

    # TTS
    CARTESIA_API_KEY: str


env = Settings()  # type: ignore
