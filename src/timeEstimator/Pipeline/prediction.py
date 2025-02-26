from timeEstimator.Config.configuration import ConfigurationManager
import cv2
import numpy as np
from timeEstimator.model.metrics import *
from timeEstimator.model.loss import *
from tensorflow.keras import models



class Prediction:
    def __init__(self):
        config = ConfigurationManager()
        self.config = config.prediction()

    def predict_time(self, model, img_path, img_size=(224, 224)):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img)[0]
        hour = (pred[0] * 12) % 12
        minute = (pred[1] * 60) % 60
        second = (pred[2] * 60) % 60

        return hour, minute, second
    
    def load_model(self, path):
        return models.load_model(
            path,
            custom_objects={
                'HourMAE': HourMAE,
                'MinuteMAE': MinuteMAE,
                'SecondMAE': SecondMAE,
                'cyclic_time_loss': cyclic_time_loss
            }
        )
    
    def start_prediction(self, img_path):
        model = self.load_model(path=self.config.model_dir)

        hr, mi, se = self.predict_time(model, img_path)

        return hr, mi, se