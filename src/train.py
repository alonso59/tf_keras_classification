__author__ = ["Germán Pinedo"]
__copyright__ = "Copyright 2021, Germán Pinedo - CINVESTAV UNIDAD GUADALAJARA"
__credits__ = ["German Pinedo"]
__license__ = "GPL"
__version__ = "0.0.1"
__maintainer__ = ["German Pinedo"]
__email__ = "german.pinedo@cinvestav.mx"
__status__ = "Released"
import yaml
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers

from networks.QRetiNet import *
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import datetime
import os
from keras.callbacks import TensorBoard, EarlyStopping
from models import ClassificationModels
import logging
from models import ClassificationModels
import sys
from keras.callbacks import Callback

def main(cfg):
    
    paths = cfg['paths']
    hyper = cfg['hyperparameters']
    general = cfg['general']
    logger, checkpoint_path, version = initialize(cfg)
    """ 
    Hyperparameters 
    """
    batch_size = hyper['batch_size']
    num_epochs = hyper['num_epochs']
    lr = hyper['lr']
    B1 = hyper['b1']
    B2 = hyper['b2']
    weight_decay = hyper['weight_decay']
    """
    Paths
    """
    train_imgdir = paths['train_imgdir']
    val_imgdir = paths['val_imgdir']
    """
    General settings
    """
    n_classes = general['n_classes']
    pretrain = general['pretrain']
    name_model = cfg['model_name']

    files_train = []
    files_val = []
    for _, _,files in os.walk(train_imgdir):
        for file in files:
                files_train.append(file)
    for _, _,files in os.walk(val_imgdir):
        for file in files:
            files_val.append(file)

    nb_train_samples = len(files_train)
    nb_validation_samples = len(files_val)

    models = ClassificationModels(n_classes=n_classes, pretrain=pretrain, logger=print)
    model = models.model_builder(name_model) # 
    _, w, h, _ = model.input_shape

    #********************************* compile ********************************************
    opt = optimizers.Adam(learning_rate=lr, beta_1=B1, beta_2=B2)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    # **************************** dataset loader generator *******************************
    train_datagen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True, vertical_flip=True, zoom_range=0.2, rotation_range=30)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_imgdir,
        target_size=(w, h),
        batch_size=batch_size,
        seed=42,
        class_mode="categorical")

    validation_generator = test_datagen.flow_from_directory(
        val_imgdir,
        seed=42,
        target_size=(w, h),
        class_mode="categorical")
    #***************************** callbacks *****************************
    checkpoint = ModelCheckpoint(checkpoint_path + name_model + '_best.hdf5', monitor='val_loss', 
                                verbose=1, save_best_only=True,
                                save_weights_only=False, 
                                mode='auto', period=1)
    
    tensorboard_callback = TensorBoard(log_dir=version + 'tensorboard/', write_graph=False, write_images=True)
    #***************************** Fiting *****************************
    history = model.fit(
        train_generator, 
        use_multiprocessing=True, 
        workers=12,
        epochs=num_epochs,
        validation_data=validation_generator,
        verbose=1,
        shuffle=True,
        steps_per_epoch=nb_train_samples // batch_size,
        validation_steps=nb_validation_samples // batch_size,
        callbacks=[checkpoint, tensorboard_callback]
    )
    #uncoment next both lines if you don't use callbacks
    model.save(checkpoint_path + name_model + '_last.hdf5')
    model.save_weights(checkpoint_path + name_model + '_weights_last.hdf5')
    #***************************** History *****************************
    df1 = pd.DataFrame({'epoch': history.epoch,
                        'accuracy_train': history.history['accuracy'],
                        'loss_train': history.history['loss'],
                        'accuracy_val': history.history['val_accuracy'],    
                        'loss_val': history.history['val_loss']
                        })
    df1.to_csv(version + "history" + ".csv")

    # Plot training & validation accuracy values
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(name_model)
    plt.grid(color='lightgray', linestyle='-', linewidth=2)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='best')
    plt.savefig(version + name_model + '_acc.png')

    # Plot training & validation loss values
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(name_model)
    plt.grid(color='lightgray', linestyle='-', linewidth=2)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='best')
    plt.savefig(version + name_model + '_loss.png')
   


def initialize(cfg):
    """Directories"""
    now = datetime.datetime.now()
    version = 'logs/' + str(now.strftime("%Y-%m-%d_%H_%M_%S")) + '/'
    checkpoint_path = version + "checkpoints/"
    create_dir(checkpoint_path)
    with open(version + 'config.txt', 'w') as text_file:
        text_file.write(f"*** Hyperparameters ***\n")
        hyp = cfg['hyperparameters']
        text_file.write(f"Loss function: {hyp['loss_fn']}\n")
        text_file.write(f"Learning rate: {hyp['lr']}\n")
        text_file.write(f"weight_decay: {hyp['weight_decay']}\n")
        text_file.write(f"BETA1, BETA2: {hyp['b1'], hyp['b2']}\n")
        text_file.write(f"Batch size: {hyp['batch_size']}\n")
        text_file.write(f"Epochs: {hyp['num_epochs']}\n")
        gen = cfg['general']
        text_file.write(f"*** Gerneral settings ***\n")
        text_file.write(f"Pretrain: {gen['pretrain']}\n")
        text_file.write(f"Num classes: {gen['n_classes']}\n")
        text_file.write(f"Model name: {cfg['model_name']}\n")
        text_file.close()

    """logging"""

    logging.basicConfig(filename=version + "info.log",
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO,
                        )
                        
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(version + "info.log"))
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger, checkpoint_path, version

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class LoggerCallback(Callback):
    def __init__(self):
        super().__init__()
    def on_epoch_end(self, epoch, logs=None):
        print("Epoch: {}, loss: {:7.4f} - accuracy: {:7.4f}".format(epoch, logs["loss"], logs["accuracy"]))

if __name__ == '__main__':
    with open('configs/classification.yaml', "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    main(cfg)
