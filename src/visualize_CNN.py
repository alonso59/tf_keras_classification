from keras.models import load_model, Model
import keras
from keras import backend as K
import tensorflow as tf
from keras.preprocessing import image
import numpy as np
from matplotlib import pyplot as plt

def display_activation(activations, col_size, row_size, act_index):
    activation = activations[act_index]
    activation_index = 0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size * 2.5, col_size * 1.5))
    for row in range(0, row_size):
        for col in range(0, col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index])
            activation_index += 1
    plt.show()

if __name__ == '__main__':
    img_width, img_height = 256, 256

    image1 = image.load_img('/home/disenolearn/Documentos/german/CNNs/D3/test/30655_right.jpeg', target_size=(img_width, img_height, 3))
    im_arra = image.img_to_array(image1)
    test_image = np.expand_dims(im_arra, axis = 0)
    model_file = '/Q-RETINET_TINY/RetiNet_tinytiny_test_1.hdf5'

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    keras.backend.set_session(session)
    plt.figure()
    plt.imshow(image1)
    plt.waitforbuttonpress()
    plt.close()
    model_gpu = load_model(model_file)
    origin_model = model_gpu.layers[-2]
    origin_model.save('temp.hdf5')
    model = load_model('temp.hdf5')
    model.summary()
    layer_outputs = [layer.output for layer in model.layers]
    activation_model = Model(inputs = model.input, outputs=layer_outputs)
    result = activation_model.predict(test_image)
    display_activation(result, 12, 8, 1)
