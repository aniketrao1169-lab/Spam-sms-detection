import json
import os
from pathlib import Path


def ensure_dir(path: str | os.PathLike) -> None:
	"""Create directory if it does not exist."""
	Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data: dict, filepath: str | os.PathLike) -> None:
	"""Save a dictionary as a JSON file with UTF-8 encoding."""
	ensure_dir(Path(filepath).parent)
	with open(filepath, "w", encoding="utf-8") as f:
		json.dump(data, f, indent=2, ensure_ascii=False)


def project_paths() -> dict:
	"""Return canonical project directories used across the pipeline."""
	root = Path(__file__).resolve().parents[1]
	return {
		"root": root,
		"data": root / "data",
		"models": root / "models",
		"results": root / "results",
	}


