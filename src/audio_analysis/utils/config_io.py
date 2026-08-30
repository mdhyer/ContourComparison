from __future__ import annotations
from pathlib import Path
from audio_analysis.config import PipelineConfig

def save_config(config: PipelineConfig, path: Path) -> None:
    """Serialize PipelineConfig to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(config.model_dump_json(indent=2))

def load_config(path: Path) -> PipelineConfig:
    """Deserialize JSON to PipelineConfig."""
    return PipelineConfig.model_validate_json(path.read_text())
