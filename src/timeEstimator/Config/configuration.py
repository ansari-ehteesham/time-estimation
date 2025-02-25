from timeEstimator.constant import *
from timeEstimator.Utils.common import create_directory, read_yaml
from timeEstimator.entity import (DataIngestionEntity)

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