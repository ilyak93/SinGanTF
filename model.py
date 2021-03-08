import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv2D, ReLU, LeakyReLU
from tensorflow.keras.layers import Add, Lambda, ZeroPadding2D

import tensorflow as tf

def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


# Xavier : tf_contrib.layers.xavier_initializer()
# He : tf_contrib.layers.variance_scaling_initializer()
# Normal : tf.random_normal_initializer(mean=0.0, stddev=0.02)
# l2_decay : tf_contrib.layers.l2_regularizer(0.0001)

weight_init = tf.keras.initializers.GlorotNormal()
weight_regularizer = None
weight_regularizer_fully = None

def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False):
    if pad > 0:
        h = x.get_shape().as_list()[1]
        if h % stride == 0:
            pad = pad * 2
        else:
            pad = max(kernel - (h % stride), 0)

        pad_top = pad // 2
        pad_bottom = pad - pad_top
        pad_left = pad // 2
        pad_right = pad - pad_left

        if pad_type == 'zero':
            x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
        if pad_type == 'reflect':
            x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')

    if sn:
        w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                            regularizer=weight_regularizer)
        x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
                         strides=[1, stride, stride, 1], padding='VALID')
        if use_bias:
            bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
            x = tf.nn.bias_add(x, bias)

    else:
        x = tf.compat.v1.layers.conv2d(inputs=x, filters=channels,
                             kernel_size=kernel, kernel_initializer=weight_init,
                             kernel_regularizer=weight_regularizer,
                             strides=stride, use_bias=use_bias)

    return x

def hw_flatten(x):
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])

def max_pooling(x) :
    return tf.compat.v1.layers.max_pooling2d(x, pool_size=2, strides=2, padding='SAME')

def attention(x, channels):
    f = conv(x, channels // 8, kernel=1, stride=1, sn=False)  # [bs, h, w, c']
    g = conv(x, channels // 8, kernel=1, stride=1, sn=False)  # [bs, h, w, c']
    h = conv(x, channels, kernel=1, stride=1, sn=False)  # [bs, h, w, c]

    # N = h * w
    s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

    beta = tf.nn.softmax(s)  # attention map

    o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
    gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

    o = tf.reshape(o, shape=x.shape)  # [bs, h, w, C]
    o = conv(o, channels, kernel=1, stride=1, sn=False)

    x = gamma * o + x

    return x

def google_attention(x, channels):
  
    batch_size, height, width, num_channels = x.get_shape().as_list()
    f = conv(x, channels // 8, kernel=1, stride=1, sn=False)  # [bs, h, w, c']
    f = max_pooling(f)

    g = conv(x, channels // 8, kernel=1, stride=1, sn=False)  # [bs, h, w, c']

    h = conv(x, channels // 2, kernel=1, stride=1, sn=False)  # [bs, h, w, c]
    h = max_pooling(h)

    # N = h * w
    s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

    beta = tf.nn.softmax(s)  # attention map

    o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
    gamma = tf.compat.v1.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

    o = tf.reshape(o, shape=[batch_size, height, width, num_channels // 2])  # [bs, h, w, C]
    o = conv(o, channels, kernel=1, stride=1, sn=False)
    x = gamma * o + x

    return x


class instance_norm(Layer):
    def __init__(self, epsilon=1e-8):
        super(instance_norm, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.beta = tf.Variable(tf.zeros([input_shape[3]]))
        self.gamma = tf.Variable(tf.ones([input_shape[3]]))

    def call(self, inputs):
        mean, var = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        x = tf.divide(tf.subtract(inputs, mean), tf.sqrt(tf.add(var, self.epsilon)))
        
        return self.gamma * x + self.beta
        

class ConvBlock(Layer):
    def __init__(self, num_filters):
        super(ConvBlock, self).__init__()
        self.num_filters = num_filters
        self.initializer = tf.random_normal_initializer(0., 0.02)
        self.conv2D = Conv2D(filters=self.num_filters,
                             kernel_size=3,
                             strides=1,
                             padding='valid',
                             use_bias=False,
                             kernel_initializer=self.initializer)
        self.instance_norm = instance_norm()

    def call(self, x):
        x = self.conv2D(x)
        x = self.instance_norm(x)
        x = LeakyReLU(alpha=0.2)(x)

        return x


class Generator(Model):
    def __init__(self, num_filters, name='Generator'):
        super(Generator, self).__init__()
        self.initializer = tf.random_normal_initializer(0., 0.02)
        self.padding = ZeroPadding2D(5)
        self.head = ConvBlock(num_filters)
        self.convblock1 = ConvBlock(num_filters)
        self.convblock2 = ConvBlock(num_filters)
        self.convblock3 = ConvBlock(num_filters)
        self.tail = Conv2D(filters=3,
                           kernel_size=3,
                           strides=1,
                           padding='valid',
                           activation='tanh',
                           kernel_initializer=self.initializer)

    def call(self, prev, noise):
        prev_pad = self.padding(prev)
        noise_pad = self.padding(noise)
        x = Add()([prev_pad, noise_pad])
        x = self.head(x)
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        _, _, _, ch = x.get_shape().as_list()
        x = google_attention(x, ch)
        x = self.tail(x)
        x = Add()([x, prev])

        return x


class Discriminator(Model):
    def __init__(self, num_filters, name='Discriminator'):
        super(Discriminator, self).__init__()
        self.initializer = tf.random_normal_initializer(0., 0.02)
        self.head = ConvBlock(num_filters)
        self.convblock1 = ConvBlock(num_filters)
        self.convblock2 = ConvBlock(num_filters)
        self.convblock3 = ConvBlock(num_filters)
        self.tail = Conv2D(filters=1,
                           kernel_size=3,
                           strides=1,
                           padding='valid',
                           kernel_initializer=self.initializer)

    def call(self, x):
        x = self.head(x)
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        _, _, _, ch = x.get_shape().as_list()
        x = google_attention(x,ch)
        x = self.tail(x)

        return x