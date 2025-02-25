from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionEntity:
    root_dir: Path
    zip_file: Path

@dataclass(frozen=True)
class DataPreProcessingEntity:
    root_dir: Path
    img_dir: Path
    label_csv: Path
    train_dir: Path
    test_dir: Path
    val_dir: Path
    data_mean: float
    data_std: float
    img_height: int
    img_width: int