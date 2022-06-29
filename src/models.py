from keras.models import Model
from networks import QRetiNet, Inception_v4, Inception_v3, ResNet50, vgg19
import tensorflow as tf

class ClassificationModels(Model):
    def __init__(self, n_classes=2, pretrain=True, dropout_prob=0.0, logger=print):
        super().__init__()
        self.n_classes = n_classes
        self.pretrain = pretrain
        self.dropout_prob = dropout_prob
        self.logger = logger
    def model_builder(self, model_name):
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope(): 
            if model_name == 'qretinet':
                self.model = self.qretinet()
            if model_name == 'inceptionv4':
                self.model = self.inceptionv4()
            if model_name == 'inceptionv3':
                self.model = self.inceptionv3()
            if model_name == 'resnet50':
                self.model = self.resnet50()
            if model_name == 'vgg19':
                self.model = self.vgg19()
        return self.model

    def qretinet(self):
        model = QRetiNet.QRetiNetv1(imgzs=224, n_classes=self.n_classes, dropout_prob=self.dropout_prob)
        model.model_summary(print_fn=self.logger)
        return model

    def inceptionv4(self):
        model_class = Inception_v4.Inceptionv4(imgzs=299, n_classes=self.n_classes, dropout_prob=self.dropout_prob)
        model = model_class.model_constructor()
        model.summary(expand_nested=True, print_fn=self.logger)
        return model

    def inceptionv3(self):
        model_class = Inception_v3.Inceptionv3(imgzs=256, n_classes=self.n_classes, dropout_prob=self.dropout_prob)
        model = model_class.model_constructor()
        model.summary(expand_nested=True, print_fn=self.logger)
        return model

    def resnet50(self):
        model_class = ResNet50.ResNet50(imgzs=256, n_classes=self.n_classes, dropout_prob=self.dropout_prob)
        model = model_class.model_constructor()
        model.summary(expand_nested=True, print_fn=self.logger)
        return model

    def vgg19(self):
        model_class = vgg19.VGG19(imgzs=256, n_classes=self.n_classes, dropout_prob=self.dropout_prob)
        model = model_class.model_constructor()
        model.summary(expand_nested=True, print_fn=self.logger)
        return model

def test():
    models = ClassificationModels(299, 2)
    model = models.model_builder('inceptionv4')

if __name__ == '__main__':
    test()