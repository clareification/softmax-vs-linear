import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

def cnn_fn(x, output_dim=10, trainable=True, group=None, num_filters=64, num_logits=1):
    """t
    Adapted from https://www.tensorflow.org/tutorials/layers
    """
    input_shape = x.shape[1:4]
    
    conv1 = tf.layers.conv2d(
          inputs=x,
          filters=num_filters,
          kernel_size=[5, 5],
          padding="same",
          activation=None,
          trainable=trainable)
    tf.add_to_collection('conv_output1', conv1)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    # pool1 = layers.Dropout(0.25)(pool1)# pool1 = tf.Print(pool1, [pool1], "Here's pooling: ")
    conv2 = tf.layers.conv2d(
          inputs=pool1,
          filters=num_filters,
          kernel_size=[5, 5],
          padding="same",
          activation=None,
          trainable=trainable)
    tf.add_to_collection('conv_output2', conv2)
    
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)
   
    # pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 8])
    pool2_flat = tf.layers.flatten(pool2)
    u = pool2_flat 
    u = tf.layers.dense(inputs=pool2_flat, units=60, activation=tf.nn.relu, trainable=trainable)
    return u

class Model():
    def __init__(self, train_features=True, fn_type='softmax', max_output=10, num_logits=1):
        self.type = fn_type
        self.output_dim = 10
        self.train_features=train_features
        self.max_output = max_output
        self.num_logits = num_logits
        self.network = self.build_network()
        self.support = np.linspace(0, 2, 51)
        self.support = self.support.astype(np.float32)
        self.batch_size = 32



    def build_network(self):
        use_softmax = self.type=='softmax'
        def f(x):
            with tf.variable_scope('features', reuse=tf.AUTO_REUSE):
                features = cnn_fn(x, trainable=self.train_features)
            with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
                if use_softmax:
                    logits = tf.layers.dense(features, units=self.output_dim*self.num_logits)
                    logits = tf.reshape(logits, [-1, self.output_dim, self.num_logits])
                    return tf.reshape(tf.matmul(tf.reshape(tf.nn.softmax(logits),
                        [-1, self.num_logits]), tf.reshape(self.support, [-1, 1])), [self.batch_size, self.output_dim])
                else:
                    return tf.layers.dense(features, self.output_dim)
        return f


class ExperimentRunner():
    def __init__(self, use_softmax, train_steps, train_features, create_model_fn, checkpoint_dir=None, batch_size=32, load_features_dir=None):

        self.train_features = train_features
        self.train_steps = train_steps
        fn_type = 'softmax' if use_softmax else None
        self.use_softmax = use_softmax
        self.ckpt_dir = checkpoint_dir
        self.model = create_model_fn(train_features, use_softmax)
        self._sess = tf.Session()
        self.checkpoint_dir = checkpoint_dir
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = (x_train/255.0).astype(np.float32)
        x_test = (x_test/255.0).astype(np.float32)
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)
        self.x_train, self.y_train, self.x_test, self.y_test = (np.reshape(x_train, list(x_train.shape) + [1]),
            y_train, x_test, y_test)
        print(x_train[0], type(x_train), x_train.dtype)
        self.batch_size = batch_size

        # Instantiate variables for graph


        if load_features_dir:
            input_ph = tf.placeholder(dtype=tf.float32, shape=[batch_size] + list(x_train.shape[1:]) + [1])

            output_ph = self.model.network(input_ph)
            self._saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='features'))
            self._saver.restore(self._sess, load_features_dir)
        if use_softmax:
            self.loss_fn = tf.nn.softmax_cross_entropy_with_logits_v2
        else:
            self.loss_fn = tf.losses.mean_squared_error
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

    
    def run_experiment(self):
        steps = self.train_steps
        num_indices = self.y_train.shape[0]
        train_indices = tf.random_uniform(
        [self.batch_size], 0, num_indices, tf.int64)
        train_labels_node = tf.gather(self.y_train, train_indices)
        train_data_node = tf.gather(self.x_train, train_indices)
        predictions = self.model.network(train_data_node)

        loss_op = self.loss_fn(train_labels_node, predictions)
        train_op = self.optimizer.minimize(loss_op)
        tf.summary.scalar('Training_loss', loss_op)
        final_loss = slim.learning.train(
          train_op,
          logdir=self.ckpt_dir,
          number_of_steps=steps,
          save_summaries_secs=1,
          log_every_n_steps=500)
        print('final loss: ', final_loss)


def model_fn(train_features, use_softmax):
    softmax_fn = 'softmax' if use_softmax else None
    num_logits = 51 if use_softmax else 1
    m = Model(train_features, softmax_fn, num_logits=num_logits)
    return m

runner = ExperimentRunner(True, 1000, True, model_fn, None)
runner.run_experiment()




