__author__ = ["Germán Pinedo"]
__copyright__ = "Copyright 2022, Germán Pinedo - CINVESTAV UNIDAD GUADALAJARA"
__credits__ = ["German Pinedo"]
__license__ = "GPL"
__version__ = "0.0.1"
__maintainer__ = ["German Pinedo"]
__email__ = "german.pinedo@cinvestav.mx"
__status__ = "Beta"

from keras.layers import Input, Dense, Flatten, MaxPooling2D, Conv2D, BatchNormalization, Dropout
from keras.models import Model
from contextlib import redirect_stdout

class QRetiNetv1(Model):
    def __init__(self, imgzs, n_classes, dropout_prob):
        super().__init__()
        self.convIn = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same', name='convIn')
        self.maxpoolI = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='mxpI')
        self.conv1A = Conv2D(filters=64, kernel_size=(5, 5), strides = (1, 1), activation='relu', padding='same', name='conv1a')
        self.conv2A = Conv2D(filters=64, kernel_size=(5, 5), strides = (1, 1), activation='relu', padding='same', name='conv2a')
        self.maxpoolA = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='mxpA')
        self.conv1B = Conv2D(filters=128, kernel_size=(3, 3), strides = (1, 1), activation='relu', padding='same', name='conv1b')
        self.conv2B = Conv2D(filters=64, kernel_size=(3, 3), strides = (1, 1), activation='relu', padding='same', name='conv2b')
        self.maxpoolB = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='mxpB')
        self.conv1C = Conv2D(filters=32, kernel_size=(3, 3), strides = (1, 1), activation='relu', padding='same', name='conv1c')
        self.conv2C = Conv2D(filters=32, kernel_size=(3, 3), strides = (1, 1), activation='relu', padding='same', name='conv2c')
        self.maxpoolC = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='mxpC')
        self.convOut = Conv2D(filters=16, kernel_size=(3, 3), strides = (1, 1), activation='relu', padding='same', name='convOut')
        self.maxpoolOut = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='mxpOut')
        self.n_classes = n_classes
        self.dropout_prob = dropout_prob
        self.imgzs = imgzs
    @property
    def __name__(self):
        return "QRetiNetv1"

    def call(self, inputs):
        x = self.convIn(inputs)
        x = self.maxpoolI(x)
        x = BatchNormalization(name='bn1')(x)
        x = self.conv1A(x)
        x = self.conv2A(x)
        x = self.maxpoolA(x)
        x = BatchNormalization(name='bn2')(x)
        x = self.conv1B(x)
        x = self.conv2B(x)
        x = self.maxpoolB(x)
        x = BatchNormalization(name='bn3')(x)
        x = self.conv1C(x)
        x = self.conv2C(x)
        x = self.maxpoolC(x)
        x = BatchNormalization(name='bn4')(x)
        x = self.convOut(x)
        x = self.maxpoolOut(x)
        x = BatchNormalization(name='bn5')(x)
        x = Flatten()(x)
        x = Dense(512, activation="relu")(x)
        # x = Dropout(self.dropout_prob)
        if self.n_classes > 1:
            x = Dense(self.n_classes, activation="softmax")(x)
        else:
            x = Dense(self.n_classes, activation="sigmoid")(x)
        return x

    def model_summary(self, print_fn):
        inputs = Input(shape=(self.imgzs, self.imgzs, 3))
        outputs = self.call(inputs)
        Model(inputs=inputs, outputs=outputs, name=self.__name__).summary(print_fn=print_fn)
