
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns
import os
from time import time
import pandas as pd
from sklearn.metrics import roc_curve, auc
import yaml

def main(cfg):
  paths = cfg['paths']
  """
  Paths
  """
  test_dir = paths['test_imgdir']

  model_file = 'logs/2022-06-29_03_27_08/checkpoints/inceptionv4_best.hdf5'

  Name_model = os.path.join(os.path.split(model_file)[0], cfg['model_name'])

  model = load_model(model_file)
  _, w, h, _ = model.input.shape
  batch_size=1

  test_datagen = ImageDataGenerator(rescale = 1. / 255)

  test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(w, h),
    shuffle=False,
    class_mode='categorical',
    batch_size=batch_size
  )

  filenames = test_generator.filenames
  nb_samples = len(filenames)
  start = time()
  predict = model.predict_generator(test_generator, steps=nb_samples//batch_size)
  stop = time()
  print(stop - start)
  y_pred = np.argmax(predict, axis=-1)
  prob_cls = predict[:,1]

  confusion_mtx = confusion_matrix(test_generator.classes, y_pred)
  sns.set(style="white")
  plt.figure()
  sns.heatmap(confusion_mtx, annot=True, fmt='',annot_kws={"size": 14})
  plt.title('Confusion Matrix ' + Name_model)
  plt.legend(loc='best')
  plt.savefig(Name_model + '_CM' + '.png')

  fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_generator.classes, prob_cls)
  auc_keras = auc(fpr_keras, tpr_keras)
  plt.figure()
  plt.plot([0, 1], [0, 1], 'k--')
  plt.plot(fpr_keras, tpr_keras, label='AUC = {:.3f}'.format(auc_keras))
  plt.xlabel('False positive rate')
  plt.ylabel('True positive rate')
  plt.title('ROC curve')
  plt.legend(loc='best')
  plt.savefig(Name_model + '_ROC-AUC' + '.png')

  df1 = pd.DataFrame({'File': test_generator.filenames,
                      'Manual': test_generator.classes,
                      'Predict': y_pred})
  df1.to_csv(Name_model + '_pred' + '.csv')


if __name__ == '__main__':
  with open('configs/classification.yaml', "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
  main(cfg)
  