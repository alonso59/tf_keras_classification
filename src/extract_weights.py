from keras.models import load_model
import numpy as np
import pandas as pd
from keras import Sequential

model = load_model('/trained_test/RetiNet_tiny_D2.hdf5')
model.load_weights('/home/disenolearn/Documentos/german/CNNs/RetiNet_tiny_D2_weights.h5')
model.summary()

f = open("../Q-RetiNet_Tiny_train/weights.h", "w")
weights = []
LayerNum = []
names = []
for layer in model.layers:
    if len(layer.weights) > 0:
        names.append(layer.name)
        weights.append(layer.get_weights()[0])
        #print(layer.name, layer.weights[0].shape)
        #print(layer.name, layer.weights[1].shape)
        #print(weights)

for (LayerNum, weights), name in zip(enumerate(weights), names):
    #print(weights.shape)
    #print(f'{name}{weights.shape}= {weights}')
    temp = np.asarray(weights)
    #print(temp.shape)
    try:
        transpose = np.transpose(temp, (3, 0, 1, 2))
        print(transpose.shape)
        f.write('\n')
        f.write(f'{name}{transpose.shape}= ')
        f.write('{')
        for i in range(transpose.shape[0]):
            f.write('{')
            for j in range(transpose.shape[1]):
                f.write('{')
                for k in range(transpose.shape[2]):
                    f.write('{')
                    for l in range(transpose.shape[3]):
                        f.write(str(transpose[i,j,k,l]))
                        f.write(',')
                    f.write('}\n')
                f.write('}')
            f.write('}')
        f.write('}')
    except:
        try:
            print(temp.shape)
            f.write('\n')
            f.write(f'{name}{temp.shape}= ')
            f.write('{')
            for i in range(temp.shape[0]):
                f.write('{')
                for j in range(temp.shape[1]):
                    f.write(str(temp[i, j]))
                    f.write(',')
                f.write('}\n')
            f.write('}')
        except:
            continue

