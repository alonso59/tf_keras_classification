from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from contextlib import redirect_stdout
from tensorflow import keras
import tensorflow as tf



def get_model_memory_usage(batch_size, model):
  import numpy as np
  from keras import backend as K

  shapes_mem_count = 0
  internal_model_mem_count = 0
  for l in model.layers:
    layer_type = l.__class__.__name__
    if layer_type == 'Model':
      internal_model_mem_count += get_model_memory_usage(batch_size, l)
    single_layer_mem = 1
    for s in l.output_shape:
      if s is None:
        continue
      single_layer_mem *= s
    shapes_mem_count += single_layer_mem

  trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
  non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

  number_size = 4.0
  if K.floatx() == 'float16':
    number_size = 2.0
  if K.floatx() == 'float64':
    number_size = 8.0

  total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
  gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
  return gbytes


model_file = 'hdf5/Inception_v4_D3'
Name_model = 'ResNet50_D3'
eval_data_dir = 'D3/test'
batch_size = 1

eval_data = ImageDataGenerator(rescale=1./255)

model = load_model(model_file+'.hdf5')
_, w, h, _ = model.input.shape
eval_generator = eval_data.flow_from_directory(
  eval_data_dir,
  target_size = (w, h),
  batch_size = batch_size,
  class_mode = "categorical",
)
print(get_model_memory_usage(batch_size, model))

with open(Name_model+'.txt', 'w') as f:
  with redirect_stdout(f):
    score = model.evaluate(eval_generator)
    print(score)



