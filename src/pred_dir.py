from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
import keras
import numpy as np
import os
from time import time
import pandas as pd

def predict_path(base_path, model, W, H):
  X_result = []
  list_images =[]
  for i in sorted(os.listdir(os.path.join(base_path))):
    _img = image.load_img(os.path.join(base_path,i), target_size=(W, H))
    img = image.img_to_array(_img)
    img = np.expand_dims(img, axis=0)
    img = img / 255
    y_pred = model.predict(img)
    predicted_class = np.argmax(y_pred)
    print(predicted_class)
    X_result.append(predicted_class)
    list_images.append(i)
  return X_result, list_images

if __name__ == '__main__':
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  session = tf.Session(config=config)
  keras.backend.set_session(session)

  model_file = 'hdf5/Inception_v4_D3.hdf5'
  test_dir = 'temp1'
  Name_dataset='temp1'
  img_width, img_height= 299, 299

  load = load_model(model_file)
  model = load.layers[-2]

  start = time()
  X_result, list_images = predict_path(test_dir, model, img_width, img_height)
  stop = time()

  print('[Elapsed Time]:',str(stop-start))
  df1 = pd.DataFrame({'File': list_images,
                      'Predict': X_result})
  df1.to_csv(Name_dataset + '_pred' + '.csv')
