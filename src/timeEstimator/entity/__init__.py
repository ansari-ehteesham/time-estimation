from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionEntity:
    root_dir: Path
    zip_file: Path