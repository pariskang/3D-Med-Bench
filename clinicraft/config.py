from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="CLINICRAFT_")

    # Anthropic
    anthropic_api_key: str = ""
    llm_model: str = "claude-opus-4-8"
    judge_model: str = "claude-opus-4-8"

    # Paths
    cases_dir: Path = Path("cases")
    resources_dir: Path = Path("resources")

    # Pulse Engine
    pulse_sdk_path: Path | None = None
    godot_bin: Path | None = None

    # UMLS
    umls_api_key: str = ""

    # MLflow
    mlflow_tracking_uri: str = "sqlite:///clinicraft.db"

    # Pipeline
    batch_size: int = 10
    max_concurrent: int = 5
    log_level: str = "INFO"

    @property
    def sign_lib_path(self) -> Path:
        return self.resources_dir / "sign_render_lib" / "signs.yaml"

    @property
    def findings_lib_path(self) -> Path:
        return self.resources_dir / "findings_lib" / "findings.yaml"

    @property
    def lr_db_path(self) -> Path:
        return self.resources_dir / "findings_lib" / "lr_database.json"


settings = Settings()
