import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from vis.utils import utils
from vis.visualization import visualize_cam
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', help='Source image path', required=True)
parser.add_argument('-m', help='Model file name', required=True)
parser.add_argument('-w', help='width image', required=True)
args = parser.parse_args()
img_path = args.i
model_file = args.m

W, H = int(args.w), int(args.w)

model = load_model(model_file)
#model = load.layers[-2]

layer_out = 0
for ilayer, layer in enumerate(model.layers):
    if "dense" in str(layer) or "Dense" in str(layer):
        layer_out = ilayer

_img = image.load_img(img_path, target_size=(W, H))
img = image.img_to_array(_img)
img = np.expand_dims(img, axis=0)
img = img / 255
y_pred = model.predict(img)
predicted_class = np.argmax(y_pred)
print("Predicted class: ")
print(predicted_class)

# Swap softmax with linear
model.layers[layer_out].activation = keras.activations.linear
model = utils.apply_modifications(model)
model.summary()
layer = input("Ingrese la capa convolucional a mostrar: ")
penultimate_layer_idx = utils.find_layer_idx(model, layer)
seed_input = img
grad_top1  = visualize_cam(model, layer_out, predicted_class, seed_input,
                           penultimate_layer_idx = penultimate_layer_idx,
                           backprop_modifier = None,
                           grad_modifier = None)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].imshow(_img)
axes[1].imshow(_img)
i = axes[1].imshow(grad_top1, cmap="jet", alpha=0.8)
fig.colorbar(i)
fig.savefig('grad_cam_class_'+str(predicted_class)+'.png')