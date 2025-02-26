import os,yaml
from pathlib import Path
from box import ConfigBox
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from ensure import ensure_annotations
from box.exceptions import BoxValueError
from timeEstimator.Exception.exception import CustomException, catch_ensure_errors
from timeEstimator.logging import logger



# Read YAML Files
@catch_ensure_errors
@ensure_annotations
def read_yaml(file_path: Path) -> ConfigBox:
    """
    It Reads the YAML File

    Arguments:
        file_path: YAML file path 
    
    Return:
        Data: YAML File Data as ConfigBox
    """
    try:
        with open(file=file_path) as yaml_file:
            data = yaml.safe_load(yaml_file)
            logger.info(f"Read YAML File: {file_path}")
            return ConfigBox(data)
    except BoxValueError:
        raise ValueError("YAML FIle is EMPTY")
    except Exception as e:
        raise CustomException(e)
    

# Create Directories
@catch_ensure_errors
@ensure_annotations
def create_directory(lst_dir:list):
    """
    It create directory
    Arguments:
        lst_dir: list of directory

    Return:
        None
    """
    for dir in lst_dir:
        dir = Path(dir)
        os.makedirs(dir, exist_ok=True)
        logger.info(f"Directory has been Created: {dir}")

def plot_training(history, save_dir, save_name='training_metrics.png',
                 save_format='png', dpi=300, show=True):
    """
    Plot and save training metrics visualization
    """
    # Create directory if needed
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(15, 5))

    # Loss plot
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss Curve')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    # Hour MAE plot
    plt.subplot(1, 3, 2)
    plt.plot(history.history['hour_mae'], label='Train')
    plt.plot(history.history['val_hour_mae'], label='Validation')
    plt.title('Hour MAE')
    plt.ylabel('Hours MAE')
    plt.xlabel('Epoch')
    plt.legend()

    # Minute & Second MAE plot
    plt.subplot(1, 3, 3)
    plt.plot(history.history['minute_mae'], label='Min Train')
    plt.plot(history.history['val_minute_mae'], label='Min Val')
    plt.plot(history.history['second_mae'], label='Sec Train')
    plt.plot(history.history['val_second_mae'], label='Sec Val')
    plt.title('Minute & Second MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()

    # Save the full figure
    full_path = os.path.join(save_dir, save_name)
    plt.savefig(full_path, format=save_format, dpi=dpi, bbox_inches='tight')

    # Save individual components
    components = {
        'loss': (0, 'loss_curve'),
        'hour_mae': (1, 'hour_mae'),
        'minute_second_mae': (2, 'minute_second_mae')
    }

    for key, (idx, name) in components.items():
        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(111)
        ax.set_position([0.15, 0.15, 0.75, 0.75])  # Adjust margins

        if key == 'loss':
            ax.plot(history.history['loss'], label='Train')
            ax.plot(history.history['val_loss'], label='Validation')
            ax.set_title('Loss Curve')
            ax.set_ylabel('Loss')
        elif key == 'hour_mae':
            ax.plot(history.history['hour_mae'], label='Train')
            ax.plot(history.history['val_hour_mae'], label='Validation')
            ax.set_title('Hour MAE')
            ax.set_ylabel('Hours MAE')
        else:
            ax.plot(history.history['minute_mae'], label='Min Train')
            ax.plot(history.history['val_minute_mae'], label='Min Val')
            ax.plot(history.history['second_mae'], label='Sec Train')
            ax.plot(history.history['val_second_mae'], label='Sec Val')
            ax.set_title('Minute & Second MAE')
            ax.set_ylabel('MAE')

        ax.set_xlabel('Epoch')
        ax.legend()
        plt.savefig(os.path.join(save_dir, f'{name}.{save_format}'),
                    format=save_format, dpi=dpi)
        plt.close(fig)

    if show:
        plt.show()
    else:
        plt.close()

    print(f"Saved training plots to: {save_dir}")


class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size=32, img_size=(224, 224), shuffle=True):
        self.df = df
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.n = len(df)
        self.indices = np.arange(self.n)

        # Clean paths and normalize labels
        self.image_paths = df['new_img_dir']
        self.labels = df[['hour', 'minute', 'second']].values / [12, 60, 60]

        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(self.n / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        X = np.empty((len(batch_indices), *self.img_size, 3), dtype=np.float32)
        y = np.empty((len(batch_indices), 3), dtype=np.float32)

        for i, idx in enumerate(batch_indices):
            X[i] = self._load_image(self.image_paths[idx])
            y[i] = self.labels[idx]

        return X, y

    def _load_image(self, path: str) -> np.ndarray:
        try:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size)
            return img.astype(np.float32) / 255.0
        except:
            return np.zeros((*self.img_size, 3), dtype=np.float32)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
