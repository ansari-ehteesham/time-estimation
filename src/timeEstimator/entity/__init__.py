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

@dataclass(frozen=True)
class ModelTrainingEntity:
    root_dir: Path
    train_csv: Path
    val_csv: Path
    model_performance: Path
    final_model: Path
    checkpoint_model: Path
    img_height: int
    img_width: int
    learning_rate: float
    epochs: int
    batch_size: int

@dataclass(frozen=True)
class PredictionEntity:
    model_dir: Path