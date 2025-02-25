from timeEstimator.logging import logger
from timeEstimator.Pipeline.stage_01 import DataIngestionPipeline

STAGE_NAME = "Data Ingestion"

logger.info(f">>>>>> STAGE {STAGE_NAME} STARTED <<<<<<") 
data_ingestion = DataIngestionPipeline()
data_ingestion.main()
logger.info(f">>>>>> STAGE {STAGE_NAME} COMPLETED <<<<<<\n\nx==========x")