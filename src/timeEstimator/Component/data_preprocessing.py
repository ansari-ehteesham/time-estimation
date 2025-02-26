import cv2, os
import numpy as np
import pandas as pd
from timeEstimator.logging import logger
from sklearn.model_selection import train_test_split
from albumentations import Compose, Resize, Normalize
from timeEstimator.entity import DataPreProcessingEntity
from timeEstimator.Exception.exception import CustomException


class DataPreProcessing:
    def __init__(self, config : DataPreProcessingEntity):
        self.config = config

    def read_img(self, img_dir):
        image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
        image = np.repeat(image, 3,axis=-1)
        resized_img = cv2.resize(image, (self.config.img_height, self.config.img_width), cv2.INTER_AREA)
        normalized_img = resized_img/255.0

        return normalized_img


    def preprocess(self, x, root_dir):

        img_name = x['img_dir']
        img_dir = os.path.join(self.config.img_dir, img_name)
        img_load = self.read_img(img_dir)
        img_uint8 = (img_load * 255).astype(np.uint8)

        save_dir = os.path.join(root_dir, img_name)
        cv2.imwrite(save_dir, img_uint8)

        return save_dir


    def load_data(self):
        try:
            df = pd.read_csv(self.config.label_csv)
            logger.info(f"Label CSV File Loaded: {self.config.label_csv}")
            
            train, test = train_test_split(df, test_size=0.15, random_state=42)
            train, val = train_test_split(train, test_size=0.15, random_state=42)

            logger.info(f"Data Splitted: Train ({len(train)}), Test ({len(test)}), Validation ({len(val)})")

            train["new_img_dir"] = train.apply(lambda x: self.preprocess(x, self.config.train_dir),axis=1)
            test["new_img_dir"] = test.apply(lambda x: self.preprocess(x, self.config.test_dir), axis=1)
            val["new_img_dir"] = val.apply(lambda x: self.preprocess(x, self.config.val_dir), axis=1)

            logger.info(f"Data Pre-Processing Completed")

            train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
            test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)
            val.to_csv(os.path.join(self.config.root_dir, "val.csv"), index=False)

            logger.info(f"Train, Test and Validation Pre-processed Data saved at: {self.config.root_dir} ")
        except Exception as e:
            raise CustomException(str(e))

        