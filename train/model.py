import tensorflow as tf


class MLPv1:

    def __init__(self, X: tf.placeholder, num_classes: int, frame_size: None, learning_rate=0.001) -> None:
        state_length = X.get_shape().as_list()[1]
        self.X = tf.reshape(X, [-1, state_length])

        self.num_classes = num_classes
        self.learning_rate = learning_rate

    def build_network(self) -> None:
        net = self.X
        net = tf.layers.dense(net, 16, activation=tf.nn.relu)
        net = tf.layers.dense(net, 64, activation=tf.nn.relu)
        net = tf.layers.dense(net, 32, activation=tf.nn.relu)
        net = tf.layers.dense(net, self.num_classes)

        self.inference = net
        self.predict = tf.argmax(self.inference, 1)

        self.Y = tf.placeholder(tf.float32, shape=[None, self.num_classes])
        self.loss = tf.losses.mean_squared_error(self.Y, self.inference)

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.loss)


class Hojoon_Custom_MLPv0:

    def __init__(self, X: tf.placeholder, num_classes: int, frame_size: None, learning_rate=0.001) -> None:
        state_length = X.get_shape().as_list()[1]
        self.X = tf.reshape(X, [-1, state_length])

        self.num_classes = num_classes
        self.learning_rate = learning_rate

    def build_network(self) -> None:
        net = self.X
        net = tf.layers.dense(net, 32, activation=tf.nn.relu)
        net = tf.layers.dense(net, 128, activation=tf.nn.relu)
        net = tf.layers.dense(net, 64, activation=tf.nn.relu)
        net = tf.layers.dense(net, self.num_classes)

        self.inference = net
        self.predict = tf.argmax(self.inference, 1)

        self.Y = tf.placeholder(tf.float32, shape=[None, self.num_classes])
        self.loss = tf.losses.mean_squared_error(self.Y, self.inference)

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.loss)


class Hojoon_Custom_CNNv0:
    def __init__(self, X: tf.placeholder, num_classes: int, frame_size: int = 1, learning_rate=0.001) -> None:
        self.X = tf.reshape(X, [-1, 32, frame_size])
        self.num_classes = num_classes
        self.learning_rate = learning_rate

    def build_network(self) -> None:
        conv1 = tf.layers.conv1d(
            self.X, 16, kernel_size=3, padding="same", activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2)
        conv2 = tf.layers.conv1d(
            pool1, 32, kernel_size=3, padding="same", activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2)
        conv3 = tf.layers.conv1d(
            pool2, 64, kernel_size=3, padding="same", activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2)
        pool3_flat = tf.reshape(pool3, [-1, 4 * 64])

        net = tf.layers.dense(pool3_flat, 256)
        net = tf.layers.dense(net, 32)
        net = tf.layers.dense(net, self.num_classes)

        self.inference = net
        self.predict = tf.argmax(self.inference, 1)

        self.Y = tf.placeholder(tf.float32, shape=[None, self.num_classes])
        self.loss = tf.losses.mean_squared_error(self.Y, self.inference)

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.loss)


class Hojoon_Custom_CNNv1:
    def __init__(self, X: tf.placeholder, num_classes: int, frame_size: int = 1, learning_rate=0.001) -> None:
        self.X = tf.reshape(X, [-1, 32, frame_size, 1])
        self.num_classes = num_classes
        self.learning_rate = learning_rate

    def build_network(self) -> None:
        conv1 = tf.layers.conv2d(inputs=self.X, filters=16, kernel_size=[
                                 3, 3], padding='SAME', activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[
                                        2, 2], strides=2, padding="SAME")
        conv2 = tf.layers.conv2d(inputs=pool1, filters=32, kernel_size=[
                                 3, 3], padding='SAME', activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[
                                        2, 2], strides=2, padding="SAME")
        conv3 = tf.layers.conv2d(inputs=pool2, filters=64, kernel_size=[
                                 3, 3], padding='SAME', activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[
                                        2, 2], strides=2, padding="SAME")
        pool3_flat = tf.reshape(pool3, [-1, 8 * 64])

        net = tf.layers.dense(pool3_flat, 256)
        net = tf.layers.dense(net, 64)
        net = tf.layers.dense(net, self.num_classes)

        self.inference = net
        self.predict = tf.argmax(self.inference, 1)

        self.Y = tf.placeholder(tf.float32, shape=[None, self.num_classes])
        self.loss = tf.losses.mean_squared_error(self.Y, self.inference)

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.loss)


class Hojoon_Custom_CNNv2:

    def __init__(self, X: tf.placeholder, num_classes: int, frame_size: int = 1, learning_rate=0.001) -> None:
        self.X = tf.reshape(X, [-1, 32, frame_size, 1])
        self.num_classes = num_classes
        self.learning_rate = learning_rate

    def build_network(self) -> None:
        conv1 = tf.layers.conv2d(inputs=self.X, filters=32, kernel_size=[
                                 3, 3], padding='SAME', activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[
                                        2, 2], strides=2, padding="SAME")
        conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[
                                 3, 3], padding='SAME', activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[
                                        2, 2], strides=2, padding="SAME")
        pool2_flat = tf.reshape(pool2, [-1, 8 * 4 * 64])

        net = tf.layers.dense(pool2_flat, 1024)
        net = tf.layers.dense(pool2_flat, 512)
        net = tf.layers.dense(net, self.num_classes)

        self.inference = net
        self.predict = tf.argmax(self.inference, 1)

        self.Y = tf.placeholder(tf.float32, shape=[None, self.num_classes])
        self.loss = tf.losses.mean_squared_error(self.Y, self.inference)

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.loss)


class Hojoon_Custom_CNN_forimage_v0:
    def __init__(self, X: tf.placeholder, num_classes: int, frame_size: int = 1, learning_rate=0.001) -> None:
        self.X = tf.reshape(X, [-1, 64, 64, frame_size])
        self.num_classes = num_classes
        self.learning_rate = learning_rate

    def build_network(self) -> None:
        conv1 = tf.layers.conv2d(inputs=self.X, filters=32, kernel_size=[
                                 3, 3], padding='SAME', activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[
                                        2, 2], strides=2, padding="SAME")
        conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[
                                 3, 3], padding='SAME', activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[
                                        2, 2], strides=2, padding="SAME")
        conv3 = tf.layers.conv2d(inputs=pool2, filters=64, kernel_size=[
                                 3, 3], padding='SAME', activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[
                                        2, 2], strides=2, padding="SAME")
        flat = tf.layers.flatten(pool3)
        net = tf.layers.dense(flat, 512)
        net = tf.layers.dense(net, self.num_classes)

        self.inference = net
        self.predict = tf.argmax(self.inference, 1)

        self.Y = tf.placeholder(tf.float32, shape=[None, self.num_classes])
        self.loss = tf.losses.mean_squared_error(self.Y, self.inference)

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.loss)


class Hojoon_Custom_CNN_forimage_v1:
    def __init__(self, X: tf.placeholder, num_classes: int, frame_size: int = 1, learning_rate=0.001) -> None:
        self.X = tf.reshape(X, [-1, frame_size, 84, 84])
        self.num_classes = num_classes
        self.learning_rate = learning_rate

    def build_network(self) -> None:
        conv1 = tf.layers.conv2d(inputs=self.X, filters=16, kernel_size=[
                                 8, 8], padding='SAME', activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(inputs=conv1, filters=32, kernel_size=[
                                 4, 4], padding='SAME', activation=tf.nn.relu)
        flat = tf.layers.flatten(conv2)
        net = tf.layers.dense(flat, 256)
        net = tf.layers.dense(net, self.num_classes)

        self.inference = net
        self.predict = tf.argmax(self.inference, 1)

        self.Y = tf.placeholder(tf.float32, shape=[None, self.num_classes])
        self.loss = tf.losses.mean_squared_error(self.Y, self.inference)

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.loss)


class Hojoon_Custom_CNN_forimage_v2:
    def __init__(self, X: tf.placeholder, num_classes: int, frame_size: int = 1, learning_rate=0.001) -> None:
        self.X = tf.reshape(X, [-1, frame_size, 64, 64])
        self.num_classes = num_classes
        self.learning_rate = learning_rate

    def build_network(self) -> None:
        conv1 = tf.layers.conv2d(inputs=self.X, filters=32, kernel_size=[
                                 3, 3], padding='SAME', activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[
                                        2, 2], strides=2, padding="SAME")
        conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[
                                 3, 3], padding='SAME', activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[
                                        2, 2], strides=2, padding="SAME")
        conv3 = tf.layers.conv2d(inputs=pool2, filters=64, kernel_size=[
                                 3, 3], padding='SAME', activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[
                                        2, 2], strides=2, padding="SAME")
        flat = tf.layers.flatten(pool3)
        net = tf.layers.dense(flat, 512)
        net = tf.layers.dense(net, self.num_classes)

        self.inference = net
        self.predict = tf.argmax(self.inference, 1)

        self.Y = tf.placeholder(tf.float32, shape=[None, self.num_classes])
        self.loss = tf.losses.mean_squared_error(self.Y, self.inference)

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.loss)


class Hojoon_Custom_LSTMv0:
    def __init__(self, X: tf.placeholder, num_classes: int, frame_size: int = 1, learning_rate=0.001) -> None:
        self.X = tf.reshape(X, [-1, frame_size, 32])
        self.num_classes = num_classes
        self.learning_rate = learning_rate

    def build_network(self) -> None:
        y = tf.compat.v1.keras.layers.LSTM(units=64, activation='tanh', recurrent_activation='hard_sigmoid',
                                           use_bias=True, kernel_initializer='glorot_uniform',
                                           recurrent_initializer='orthogonal', bias_initializer='zeros',
                                           dropout=0, recurrent_dropout=0, implementation=1)(self.X)
        y = tf.keras.layers.Reshape((1, 64))(y)

        y = tf.keras.layers.LSTM(units=32, activation='tanh', recurrent_activation='hard_sigmoid',
                                 use_bias=True, kernel_initializer='glorot_uniform',
                                 recurrent_initializer='orthogonal', bias_initializer='zeros',
                                 dropout=0, recurrent_dropout=0, implementation=1)(y)

        net = tf.layers.dense(y, self.num_classes)
        self.inference = net
        self.predict = tf.argmax(self.inference, 1)
        self.Y = tf.placeholder(tf.float32, shape=[None, self.num_classes])
        self.loss = tf.losses.mean_squared_error(self.Y, self.inference)
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.loss)


class Hojoon_Custom_C_LSTMv0:
    def __init__(self, X: tf.placeholder, num_classes: int, frame_size: int = 1, learning_rate=0.001) -> None:
        self.X = tf.reshape(X, [-1, 64, 64, frame_size])
        self.num_classes = num_classes
        self.learning_rate = learning_rate

    def build_network(self) -> None:
        conv1 = tf.layers.conv2d(inputs=self.X, filters=32, kernel_size=[
                                 3, 3], padding='SAME', activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[
                                        2, 2], strides=2, padding="SAME")
        conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[
                                 3, 3], padding='SAME', activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[
                                        2, 2], strides=2, padding="SAME")
        flat = tf.layers.flatten(pool2)
        net = tf.layers.dense(flat, 512)
        net = tf.layers.dense(net, 64)

        net = tf.keras.layers.Reshape((1, 64))(net)
        y = tf.keras.layers.LSTM(units=64, activation='tanh', recurrent_activation='hard_sigmoid',
                                 use_bias=True, kernel_initializer='glorot_uniform',
                                 recurrent_initializer='orthogonal', bias_initializer='zeros',
                                 dropout=0, recurrent_dropout=0, implementation=1)(net)

        net = tf.layers.dense(y, self.num_classes)
        self.inference = net
        self.predict = tf.argmax(self.inference, 1)
        self.Y = tf.placeholder(tf.float32, shape=[None, self.num_classes])
        self.loss = tf.losses.mean_squared_error(self.Y, self.inference)
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.loss)


class ConvNetv1:

    def __init__(self, X: tf.placeholder, num_classes: int, frame_size: int = 1, learning_rate=0.001) -> None:
        self.X = tf.reshape(X, [-1, 32, frame_size])

        self.num_classes = num_classes
        self.learning_rate = learning_rate

    def build_network(self) -> None:
        conv1 = tf.layers.conv1d(
            self.X, 8, kernel_size=3, padding="same", activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2)

        conv2 = tf.layers.conv1d(
            pool1, 16, kernel_size=3, padding="same", activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2)

        conv3 = tf.layers.conv1d(
            pool2, 32, kernel_size=3, padding="same", activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2)
        pool3_flat = tf.reshape(pool3, [-1, 4 * 32])

        net = tf.layers.dense(pool3_flat, 128)
        net = tf.layers.dense(net, 32)
        net = tf.layers.dense(net, self.num_classes)

        self.inference = net
        self.predict = tf.argmax(self.inference, 1)

        self.Y = tf.placeholder(tf.float32, shape=[None, self.num_classes])
        self.loss = tf.losses.mean_squared_error(self.Y, self.inference)

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.loss)


class ConvNetv2:

    def __init__(self, X: tf.placeholder, num_classes: int, frame_size: int = 1, learning_rate=0.001) -> None:
        self.X = tf.reshape(X, [-1, 128, frame_size])

        self.num_classes = num_classes
        self.learning_rate = learning_rate

    def build_network(self) -> None:
        conv1 = tf.layers.conv1d(
            self.X, 128, kernel_size=7, padding="same", activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=4, strides=2)

        conv2 = tf.layers.conv1d(
            pool1, 256, kernel_size=5, padding="same", activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=3, strides=2)

        conv3 = tf.layers.conv1d(
            pool2, 512, kernel_size=3, padding="same", activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2)

        conv4 = tf.layers.conv1d(
            pool3, 512, kernel_size=3, padding="same", activation=tf.nn.relu)
        pool4 = tf.layers.max_pooling1d(inputs=conv4, pool_size=2, strides=2)

        conv5 = tf.layers.conv1d(
            pool4, 512, kernel_size=3, padding="same", activation=tf.nn.relu)
        pool5 = tf.layers.max_pooling1d(inputs=conv5, pool_size=2, strides=2)
        pool5_flat = tf.reshape(pool5, [-1, 3 * 512])

        net = tf.layers.dense(pool5_flat, 1024)
        net = tf.layers.dense(net, 256)
        net = tf.layers.dense(net, self.num_classes)

        self.inference = net
        self.predict = tf.argmax(self.inference, 1)

        self.Y = tf.placeholder(tf.float32, shape=[None, self.num_classes])
        self.loss = tf.losses.mean_squared_error(self.Y, self.inference)

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.loss)
