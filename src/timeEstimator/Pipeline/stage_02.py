from timeEstimator.Config.configuration import ConfigurationManager
from timeEstimator.Component.data_preprocessing import DataPreProcessing

class DataPreProcessingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        preprocess_config = config.data_preprocessing_config()
        data_preprocessing = DataPreProcessing(preprocess_config)
        data_preprocessing.load_data()