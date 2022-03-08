import tensorflow as tf
import tensorflow.keras.applications as keras_apps
from tensorflow.keras import layers, Input, Model


class MLPv1:
    def __init__(self, X: tf.compat.v1.placeholder, num_classes: int, frame_size: None, learning_rate=0.001) -> None:
        state_length = X.get_shape().as_list()[1]
        self.X = tf.compat.v1.reshape(X, [-1, state_length])

        self.num_classes = num_classes
        self.learning_rate = learning_rate

    def build_network(self) -> None:
        net = self.X
        net = tf.compat.v1.layers.dense(net, 16, activation=tf.nn.relu)
        net = tf.compat.v1.layers.dense(net, 64, activation=tf.nn.relu)
        net = tf.compat.v1.layers.dense(net, 32, activation=tf.nn.relu)
        net = tf.compat.v1.layers.dense(net, self.num_classes)

        self.inference = net
        self.predict = tf.argmax(self.inference, 1)

        self.Y = tf.compat.v1.placeholder(tf.compat.v1.float32, shape=[None, self.num_classes])
        self.loss = tf.compat.v1.losses.mean_squared_error(self.Y, self.inference)

        self.optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.loss)


class Custom_CNN_forimage_v1:
    def __init__(self, X: tf.compat.v1.placeholder, num_classes: int, frame_size: int = 1, learning_rate=0.001) -> None:
        self.X = tf.reshape(X, [-1, frame_size, 84, 84])
        self.num_classes = num_classes
        self.learning_rate = learning_rate

    def build_network(self) -> None:
        conv1 = tf.compat.v1.layers.conv2d(inputs=self.X, filters=16, kernel_size=[
            8, 8], padding='SAME', activation=tf.nn.relu)
        conv2 = tf.compat.v1.layers.conv2d(inputs=conv1, filters=32, kernel_size=[
            4, 4], padding='SAME', activation=tf.nn.relu)
        flat = tf.compat.v1.layers.flatten(conv2)
        net = tf.compat.v1.layers.dense(flat, 256)
        net = tf.compat.v1.layers.dense(net, self.num_classes)

        self.inference = net
        self.predict = tf.argmax(self.inference, 1)

        self.Y = tf.compat.v1.placeholder(tf.float32, shape=[None, self.num_classes])
        self.loss = tf.compat.v1.losses.mean_squared_error(self.Y, self.inference)

        self.optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.loss)


class Custom_CNN_forimage_v2:
    def __init__(self, X: tf.compat.v1.placeholder, num_classes: int, frame_size: int = 1, learning_rate=0.001) -> None:
        self.X = tf.reshape(X, [-1, frame_size, 64, 64])
        self.num_classes = num_classes
        self.learning_rate = learning_rate

    def build_network(self) -> None:
        conv1 = tf.compat.v1.layers.conv2d(inputs=self.X, filters=32, kernel_size=[
            3, 3], padding='SAME', activation=tf.nn.relu)
        pool1 = tf.compat.v1.layers.max_pooling2d(inputs=conv1, pool_size=[
            2, 2], strides=2, padding="SAME")
        conv2 = tf.compat.v1.layers.conv2d(inputs=pool1, filters=64, kernel_size=[
            3, 3], padding='SAME', activation=tf.nn.relu)
        pool2 = tf.compat.v1.layers.max_pooling2d(inputs=conv2, pool_size=[
            2, 2], strides=2, padding="SAME")
        conv3 = tf.compat.v1.layers.conv2d(inputs=pool2, filters=64, kernel_size=[
            3, 3], padding='SAME', activation=tf.nn.relu)
        pool3 = tf.compat.v1.layers.max_pooling2d(inputs=conv3, pool_size=[
            2, 2], strides=2, padding="SAME")
        flat = tf.compat.v1.layers.flatten(pool3)
        net = tf.compat.v1.layers.dense(flat, 512)
        net = tf.compat.v1.layers.dense(net, self.num_classes)

        self.inference = net
        self.predict = tf.argmax(self.inference, 1)

        self.Y = tf.compat.v1.placeholder(tf.float32, shape=[None, self.num_classes])
        self.loss = tf.compat.v1.losses.mean_squared_error(self.Y, self.inference)

        self.optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.loss)


def select_cnn_model(model_name, weights='imagenet', classes=1, classifier_activation='softmax'):
    args = {
        'weights': weights,
        'include_top': True
    }
    if model_name == 'VGG16':
        input_shape = (224,224,3)
        model = keras_apps.vgg16.VGG16(**args, input_shape=input_shape, classifier_activation=classifier_activation)
    if model_name == 'VGG19':
        input_shape = (224,224,3)
        model = keras_apps.vgg19.VGG19(**args, input_shape=input_shape, classifier_activation=classifier_activation)
    if model_name == 'MobileNet':
        input_shape = (224,224,3)
        model = keras_apps.mobilenet.MobileNet(**args, input_shape=input_shape, classifier_activation=classifier_activation)
    if model_name == 'MobileNetV2':
        input_shape = (224,224,3)
        model = keras_apps.mobilenet_v2.MobileNetV2(**args, input_shape=input_shape, classifier_activation=classifier_activation)
    if model_name == 'MobileNetV3Large':
        input_shape = (224,224,3)
        model = keras_apps.MobileNetV3Large(**args, input_shape=input_shape, classifier_activation=classifier_activation)
    if model_name == 'MobileNetV3Small':
        input_shape = (224,224,3)
        model = keras_apps.MobileNetV3Small(**args, input_shape=input_shape, classifier_activation=classifier_activation)
    if model_name == 'DenseNet121':
        input_shape = (224,224,3)
        model = keras_apps.densenet.DenseNet121(**args, input_shape=input_shape)
    if model_name == 'DenseNet169':
        input_shape = (224,224,3)
        model = keras_apps.densenet.DenseNet169(**args, input_shape=input_shape)
    if model_name == 'DenseNet201':
        input_shape = (224,224,3)
        model = keras_apps.densenet.DenseNet201(**args, input_shape=input_shape)
    if model_name == 'ResNet50':
        input_shape = (224,224,3)
        model = keras_apps.resnet50.ResNet50(**args, input_shape=input_shape, classifier_activation=classifier_activation)
    if model_name == 'ResNet101':
        input_shape = (224,224,3)
        model = keras_apps.resnet.ResNet101(**args, input_shape=input_shape, classifier_activation=classifier_activation)
    if model_name == 'ResNet152':
        input_shape = (224,224,3)
        model = keras_apps.resnet.ResNet152(**args, input_shape=input_shape, classifier_activation=classifier_activation)
    if model_name == 'ResNet50V2':
        input_shape = (224,224,3)
        model = keras_apps.resnet_v2.ResNet50V2(**args, input_shape=input_shape, classifier_activation=classifier_activation)
    if model_name == 'ResNet101V2':
        input_shape = (224,224,3)
        model = keras_apps.resnet_v2.ResNet101V2(**args, input_shape=input_shape, classifier_activation=classifier_activation)
    if model_name == 'ResNet152V2':
        input_shape = (224,224,3)
        model = keras_apps.resnet_v2.ResNet152V2(**args, input_shape=input_shape, classifier_activation=classifier_activation)
    if model_name == 'InceptionV3':
        input_shape = (299,299,3)
        model = keras_apps.inception_v3.InceptionV3(**args, input_shape=input_shape, classifier_activation=classifier_activation)
    if model_name == 'InceptionResNetV2':
        input_shape = (299,299,3)
        model = keras_apps.inception_resnet_v2.InceptionResNetV2(**args, input_shape=input_shape, classifier_activation=classifier_activation)
    if model_name == 'EfficientNetB0':
        input_shape = (224,224,3)
        model = keras_apps.efficientnet.EfficientNetB0(**args, input_shape=input_shape, classifier_activation=classifier_activation)
    if model_name == 'EfficientNetB1':
        input_shape = (240,240,3)
        model = keras_apps.efficientnet.EfficientNetB1(**args, input_shape=input_shape, classifier_activation=classifier_activation)
    if model_name == 'EfficientNetB2':
        input_shape = (260,260,3)
        model = keras_apps.efficientnet.EfficientNetB2(**args, classifier_activation=classifier_activation)
    if model_name == 'EfficientNetB3':
        input_shape = (300,300,3)
        model = keras_apps.efficientnet.EfficientNetB3(**args, classifier_activation=classifier_activation)
    if model_name == 'EfficientNetB4':
        input_shape = (380,380,3)
        model = keras_apps.efficientnet.EfficientNetB4(**args, classifier_activation=classifier_activation)
    if model_name == 'EfficientNetB5':
        input_shape = (456,456,3)
        model = keras_apps.efficientnet.EfficientNetB5(**args, classifier_activation=classifier_activation)
    if model_name == 'EfficientNetB6':
        input_shape = (528,528,3)
        model = keras_apps.efficientnet.EfficientNetB6(**args, classifier_activation=classifier_activation)
    if model_name == 'EfficientNetB7':
        input_shape = (600,600,3)
        model = keras_apps.efficientnet.EfficientNetB7(**args, classifier_activation=classifier_activation)
    if model_name == 'Xception':
        input_shape = (299,299,3)
        model = keras_apps.xception.Xception(**args, classifier_activation=classifier_activation)
    if model_name == 'NASNetLarge':
        input_shape = (331,331,3)
        model = keras_apps.nasnet.NASNetLarge(**args)
    if model_name == 'NASNetMobile':
        input_shape = (224,224,3)
        model = keras_apps.nasnet.NASNetMobile(**args)

    # use imagenet backbone and add custom fully connected layer
    inputs = Input(shape=input_shape)
    x = model(inputs, training=True)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(16, activation='relu')(x)
    if classes == 1:
        activation = 'sigmoid'
    else:
        activation = 'softmax'
    outputs = layers.Dense(classes, activation=activation)(x)
    output_model = Model(inputs, outputs)
    return output_model, input_shape


def test_load_cnn_models(model_list):
    for model_name in model_list:
        print(model_name)
        model, _ = select_cnn_model(model_name)
        print(model.summary())
