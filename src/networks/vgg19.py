__author__ = ["Germán Pinedo"]
__copyright__ = "Copyright 2021, Germán Pinedo - CINVESTAV UNIDAD GUADALAJARA"
__credits__ = ["German Pinedo"]
__license__ = "GPL"
__version__ = "0.0.1"
__maintainer__ = ["German Pinedo"]
__email__ = "german.pinedo@cinvestav.mx"
__status__ = "Released"

from keras.layers import Dropout, Dense, Flatten
from keras.models import Model
from keras import applications

class VGG19:
    def __init__(self, imgzs, n_classes, dropout_prob, pretrain='imagenet'):
        super().__init__()
        self.n_classes = n_classes
        self.dropout_prob = dropout_prob
        self.imgzs = imgzs
        self.pretrain = pretrain

    @property
    def __name__(self):
        return "VGG19"

    def model_constructor(self):
        base_model = applications.VGG19(include_top=False, weights='imagenet', input_shape=(self.imgzs, self.imgzs, 3))
        x = base_model.output
        x = Flatten()(x)
        x = Dense(units=256, activation='relu')(x)
        x = Dropout(self.dropout_prob)(x)
        if self.n_classes > 1:
            predictions = Dense(self.n_classes, activation="softmax")(x)
        else:
            predictions = Dense(self.n_classes, activation="sigmoid")(x)
        model = Model(inputs=base_model.input, outputs=predictions, name=self.__name__)
        return model