from timeEstimator.logging import logger
from timeEstimator.Pipeline.stage_01 import DataIngestionPipeline
from timeEstimator.Pipeline.stage_02 import DataPreProcessingPipeline
from timeEstimator.Pipeline.stage_03 import ModelTrainingPipeline

STAGE_NAME = "Data Ingestion"

logger.info(f">>>>>> STAGE {STAGE_NAME} STARTED <<<<<<") 
data_ingestion = DataIngestionPipeline()
data_ingestion.main()
logger.info(f">>>>>> STAGE {STAGE_NAME} COMPLETED <<<<<<\n\nx==========x")



STAGE_NAME = "Data Pre-Processing"

logger.info(f">>>>>> STAGE {STAGE_NAME} STARTED <<<<<<") 
data_preprocess = DataPreProcessingPipeline()
data_preprocess.main()
logger.info(f">>>>>> STAGE {STAGE_NAME} COMPLETED <<<<<<\n\nx==========x")



STAGE_NAME = "Model Training"

logger.info(f">>>>>> STAGE {STAGE_NAME} STARTED <<<<<<") 
model_training = ModelTrainingPipeline()
model_training.main()
logger.info(f">>>>>> STAGE {STAGE_NAME} COMPLETED <<<<<<\n\nx==========x")