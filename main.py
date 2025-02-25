from timeEstimator.logging import logger
from timeEstimator.Pipeline.stage_01 import DataIngestionPipeline
from timeEstimator.Pipeline.stage_02 import DataPreProcessingPipeline

STAGE_NAME = "Data Ingestion"

logger.info(f">>>>>> STAGE {STAGE_NAME} STARTED <<<<<<") 
data_ingestion = DataIngestionPipeline()
data_ingestion.main()
logger.info(f">>>>>> STAGE {STAGE_NAME} COMPLETED <<<<<<\n\nx==========x")



STAGE_NAME = "Data Pre-Processing"

logger.info(f">>>>>> STAGE {STAGE_NAME} STARTED <<<<<<") 
data_ingestion = DataPreProcessingPipeline()
data_ingestion.main()
logger.info(f">>>>>> STAGE {STAGE_NAME} COMPLETED <<<<<<\n\nx==========x")