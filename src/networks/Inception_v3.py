
from keras.layers import Dropout, Dense, Flatten
from keras.models import Model
from keras import applications


class Inceptionv3:
    def __init__(self, imgzs, n_classes, dropout_prob, pretrain='imagenet'):
        super().__init__()
        self.n_classes = n_classes
        self.dropout_prob = dropout_prob
        self.imgzs = imgzs
        self.pretrain = pretrain

    @property
    def __name__(self):
        return "InceptionV3"

    def model_constructor(self):
        base_model = applications.ResNet50(include_top=False, weights='imagenet', input_shape=(self.imgzs, self.imgzs, 3))
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