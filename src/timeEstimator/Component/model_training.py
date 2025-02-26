from timeEstimator.model.custom_resnet import create_time_model
from timeEstimator.model.metrics import HourMAE, MinuteMAE, SecondMAE
from timeEstimator.model.loss import cyclic_time_loss
from timeEstimator.Utils.common import CustomDataGenerator
from timeEstimator.entity import ModelTrainingEntity
import pandas as pd
import tensorflow as tf


class ModelTraining:
    def __init__(self, config: ModelTrainingEntity):
        self.config = config

    def train_model(self, train_gen, val_gen, img_size):
        
        model = create_time_model(img_size)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss=cyclic_time_loss,
            metrics=[HourMAE(), MinuteMAE(), SecondMAE()]
        )

        callbacks_lst = [
            tf.keras.callbacks.EarlyStopping(
                patience=15,
                monitor='val_hour_mae',
                mode='min',
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                self.config.checkpoint_model,
                monitor='val_hour_mae',
                save_best_only=True,
                mode='min',
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_hour_mae',
                factor=0.2,
                patience=5,
                mode='min',
                min_lr=1e-6
            )
        ]

        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=self.config.epochs,
            callbacks=callbacks_lst
        )

        return model, history

    def save_model(self, model, path):
        path = self.config.final_model
        model.save(path)

    def model_train(self):
        train_df = pd.read_csv("artifacts/data_preprocessed/train.csv")
        val_df = pd.read_csv("artifacts/data_preprocessed/val.csv")

        train_gen = CustomDataGenerator(df=train_df,batch_size=self.config.batch_size)
        val_gen = CustomDataGenerator(df=val_df, batch_size=self.config.batch_size,shuffle=False)

        # Train
        img_size = (self.config.img_height, self.config.img_width)
        model, history = self.train_model(train_gen, val_gen, img_size)
        self.plot_training(
            history,
            save_dir=self.config.model_performance,
            save_name='training_metrics.svg',
            save_format='svg',
            dpi=300,
            show=False
        )

        # Save
        self.save_model(model)
