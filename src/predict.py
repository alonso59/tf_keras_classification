from keras.models import load_model
from keras.preprocessing import image
import keras
import tensorflow as tf
import numpy as np
from CNN.models.RetiNet import RetiNet_model
def predict(imag, model_file):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    keras.backend.set_session(session)

    _, __, img_width, img_height = RetiNet_model("")

    _img = image.load_img(imag, target_size=(img_width, img_height))
    img = image.img_to_array(_img)
    img = np.expand_dims(img, axis=0)
    img = img / 255
    model = load_model(model_file)
    result = model.predict(img)
    pred_classes = np.argmax(result)
    return pred_classes, np.max(result)
