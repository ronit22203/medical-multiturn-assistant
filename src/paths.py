from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(path: str | Path) -> Path:
    """Resolve repository resources independently of the process working directory."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
