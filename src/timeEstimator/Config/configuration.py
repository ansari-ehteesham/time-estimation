from timeEstimator.constant import *
from timeEstimator.Utils.common import create_directory, read_yaml
from timeEstimator.entity import (DataIngestionEntity,
                                  DataPreProcessingEntity,
                                  ModelTrainingEntity,
                                  PredictionEntity)

class ConfigurationManager:
    def __init__(self, params=PARAMS_FILE_PATH, config=CONFIG_FILE_PATH):
        self.params = read_yaml(params)
        self.config = read_yaml(config)

        create_directory([self.config.root_dir])

    def data_ingestion_config(self):
        config = self.config.data_ingestion

        entity = DataIngestionEntity(
            root_dir = config.root_dir,
            zip_file = config.zip_file
        )

        return entity
    
    def data_preprocessing_config(self):
        config = self.config.data_preprocessed
        params = self.params

        create_directory([
            config.root_dir,
            config.train_dir,
            config.test_dir,
            config.val_dir
        ])

        config = DataPreProcessingEntity(
            root_dir = Path(config.root_dir),
            img_dir = Path(config.img_dir),
            label_csv = Path(config.label_csv),
            train_dir = Path(config.train_dir),
            test_dir = Path(config.test_dir),
            val_dir = Path(config.val_dir),
            data_mean = params.data_mean,
            data_std = params.data_std,
            img_height = params.img_height,
            img_width = params.img_width
        )

        return config
    
    def model_training_config(self):
        config = self.config.model_training
        params = self.params
        create_directory([config.root_dir])
        entity = ModelTrainingEntity(
            root_dir = config.root_dir,
            train_csv = config.train_csv,
            val_csv = config.val_csv,
            model_performance = config.model_performance,
            final_model = config.final_model,
            checkpoint_model = config.checkpoint_model,
            img_height = params.img_height,
            img_width = params.img_width,
            learning_rate = params.learning_rate,
            epochs = params.epochs,
            batch_size = params.batch_size
        )

        return entity
    
    def prediction(self):
        config = self.config.prediction

        entity = PredictionEntity(
            model_dir = config.model_dir
        )

        return entity