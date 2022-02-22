import tensorflow as tf


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
