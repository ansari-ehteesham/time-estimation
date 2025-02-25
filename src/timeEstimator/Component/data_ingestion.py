import os
from zipfile import ZipFile
from timeEstimator.logging import logger
from timeEstimator.Exception.exception import CustomException
from timeEstimator.entity import DataIngestionEntity

class DataIngestion:
    def __init__(self, config : DataIngestionEntity):
        self.config = config

    def rename_folder(self):
        for file in os.listdir(self.config.root_dir):   
            if "raw" in file:
                os.rename(f"{self.config.root_dir}/{file}", f"{self.config.root_dir}/data_ingestion")
                logger.info(f"File Name Changed from {file} to data_ingestion")
                break

    def data_ingest(self):
        try:
            logger.info(f"Zip Extarction Started : {self.config.zip_file}")
            with ZipFile(self.config.zip_file, 'r') as f:
                f.extractall(self.config.root_dir)
                f.close()
            self.rename_folder()
            logger.info(f"Zip Extraction Finished at : {os.path.join(self.config.root_dir, "data_ingestion")}")
        except Exception as e:
            raise CustomException(str(e))

        