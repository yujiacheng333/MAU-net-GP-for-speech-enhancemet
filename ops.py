import tensorflow as tf
import numpy as np
from tensorflow.python import keras
tf.enable_eager_execution()
class DiscDownsample(tf.keras.Model):
    def __init__(self, filters, kernel_size):
        super(DiscDownsample, self).__init__()
        self.initializer = tf.keras.initializers.random_normal(0, 0.02)
        self.conv1 = tf.keras.layers.Conv2D(filters=filters,
                                            kernel_size=kernel_size,
                                            strides=2,
                                            kernel_initializer=self.initializer,
                                            use_bias=False,
                                            padding="valid")

    def call(self, x, normal=True):
        if normal:
            x = self.conv1(x)
            x = tf.nn.leaky_relu(x)

        else:
            x = self.conv1(x)
            x = tf.nn.leaky_relu(x)
        return x


class Downsample(keras.Model):

    def __init__(self, filters, size, apply_batchnorm=True):
        super(Downsample, self).__init__()

        self.apply_batchnorm = apply_batchnorm
        initializer = tf.random_normal_initializer(0., 0.02)

        self.conv1 = keras.layers.Conv2D(filters,
                                         (size, size),
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         use_bias=False)
        self.conv2 = keras.layers.Conv2D(filters=filters,
                                         kernel_size=3,
                                         strides=1,
                                         padding="same",

                                         use_bias=False)
        if self.apply_batchnorm:
            self.batchnorm = keras.layers.BatchNormalization()

    def call(self, x, training):
        x = self.conv1(x)
        x = self.conv2(x)
        if self.apply_batchnorm:
            x = self.batchnorm(x, training=training)
        x = tf.nn.leaky_relu(x)
        return x
class Upsample(keras.Model):

    def __init__(self, filters, size, apply_dropout=False):
        super(Upsample, self).__init__()

        self.apply_dropout = apply_dropout
        initializer = tf.random_normal_initializer(0., 0.02)

        self.up_conv = keras.layers.Conv2DTranspose(filters,
                                                    (size, size),
                                                    strides=2,
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
        self.batchnorm = keras.layers.BatchNormalization()
        if self.apply_dropout:
            self.dropout = keras.layers.Dropout(0.5)

    def call(self, x1, x2, training=None):

        x = self.up_conv(x1)
        x = self.batchnorm(x, training=training)
        if self.apply_dropout:
            x = self.dropout(x, training=training)
        x = tf.nn.relu(x)
        x = tf.concat([x, x2], axis=-1)
        return x

class generator(keras.Model):

    def __init__(self):
        super(generator, self).__init__()

        initializer = tf.random_normal_initializer(0., 0.02)

        self.down1 = Downsample(64, 4, apply_batchnorm=False)
        self.down2 = Downsample(128, 4)
        self.down3 = Downsample(256, 4)
        self.down4 = Downsample(512, 4)
        self.down5 = Downsample(512, 4)
        self.down6 = Downsample(512, 4)
        self.down7 = Downsample(512, 4)
        self.down8 = Downsample(512, 4)

        self.up1 = Upsample(512, 4, apply_dropout=True)
        self.up2 = Upsample(512, 4, apply_dropout=True)
        self.up3 = Upsample(512, 4, apply_dropout=True)
        self.up4 = Upsample(512, 4)
        self.up5 = Upsample(256, 4)
        self.up6 = Upsample(128, 4)
        self.up7 = Upsample(64, 4)

        self.last = keras.layers.Conv2DTranspose(1, (4, 4),
                                                    strides=2,
                                                    padding='same',
                                                    kernel_initializer=initializer)


    def call(self, x, training=None):

        # x shape == (bs, 256, 256, 3)
        x1 = self.down1(x, training=training)  # (bs, 128, 128, 64)
        x2 = self.down2(x1, training=training)  # (bs, 64, 64, 128)
        x3 = self.down3(x2, training=training)  # (bs, 32, 32, 256)
        x4 = self.down4(x3, training=training)  # (bs, 16, 16, 512)
        x5 = self.down5(x4, training=training)  # (bs, 8, 8, 512)
        x6 = self.down6(x5, training=training)  # (bs, 4, 4, 512)
        x7 = self.down7(x6, training=training)  # (bs, 2, 2, 512)
        x8 = self.down8(x7, training=training)  # (bs, 1, 1, 512)

        x9 = self.up1(x8, x7, training=training)  # (bs, 2, 2, 1024)
        x10 = self.up2(x9, x6, training=training)  # (bs, 4, 4, 1024)
        x11 = self.up3(x10, x5, training=training)  # (bs, 8, 8, 1024)
        x12 = self.up4(x11, x4, training=training)  # (bs, 16, 16, 1024)
        x13 = self.up5(x12, x3, training=training)  # (bs, 32, 32, 512)
        x14 = self.up6(x13, x2, training=training)  # (bs, 64, 64, 256)
        x15 = self.up7(x14, x1, training=training)  # (bs, 128, 128, 128)

        x16 = self.last(x15)  # (bs, 256, 256, 3)
        x16 = tf.nn.relu(x16)

        return x16

"""
class Discriminator(keras.Model):

    def __init__(self):
        super(Discriminator, self).__init__()

        initializer = tf.random_normal_initializer(0., 0.02)

        self.down1 = DiscDownsample(64, 4, False)
        self.down2 = DiscDownsample(128, 4)
        self.down3 = DiscDownsample(256, 4)
        self.down4 = DiscDownsample(256, 4)
        self.down5 = DiscDownsample(512, 4)
        self.down6 = DiscDownsample(512, 4)
        self.down7 = DiscDownsample(512, 4)
        # we are zero padding here with 1 because we need our shape to
        # go from (batch_size, 32, 32, 256) to (batch_size, 31, 31, 512)
        self.zero_pad1 = keras.layers.ZeroPadding2D()
        self.conv = keras.layers.Conv2D(512, (4, 4),
                                           strides=1,
                                           kernel_initializer=initializer,
                                           use_bias=False)
        self.batchnorm1 = keras.layers.BatchNormalization()

        # shape change from (batch_size, 31, 31, 512) to (batch_size, 30, 30, 1)
        self.zero_pad2 = keras.layers.ZeroPadding2D()
        self.flatten = keras.layers.Flatten()
        self.last = keras.layers.Dense(1)


    def call(self, inputs, training=True):
        inp, target = inputs

        # concatenating the input and the target
        x = tf.concat([inp, target], axis=-1)  # (bs, 256, 256, channels*2)
        x = self.down1(x, training=training)  # (bs, 128, 128, 64)
        x = self.down2(x, training=training)  # (bs, 64, 64, 128)
        x = self.down3(x, training=training)  # (bs, 32, 32, 256)
        x = self.down4(x, training=training)
        x = self.down5(x, training=training)
        x = self.down7(x, training=training)
        x = self.conv(x)  # (bs, 31, 31, 512)
        x = self.batchnorm1(x, training=training)
        x = tf.nn.leaky_relu(x)
        # don't add a sigmoid activation here since
        # the loss function expects raw logits.
        x = self.flatten(x)
        x = self.last(x)  # (bs, 30, 30, 1)
        return x
        
"""
class SpectralNormalization(tf.keras.layers.Layer):
    def __init__(self, layer):
        super(SpectralNormalization, self).__init__()
        self.layer = layer

    def spectral_norm(self, w, r=5):
        w_shape = tf.keras.backend.int_shape(w)
        in_dim = np.prod(w_shape[:-1]).astype(int)
        out_dim = w_shape[-1]
        w = tf.reshape(w, (in_dim, out_dim))
        u = tf.ones((1, in_dim))
        for i in range(r):
            v = tf.nn.l2_normalize(tf.keras.backend.dot(u, w))
            u = tf.nn.l2_normalize(tf.keras.backend.dot(v, tf.transpose(w)))
        return tf.keras.backend.sum(tf.keras.backend.dot(tf.keras.backend.dot(u, w), tf.transpose(v)))

    def spectral_normalization(self, w):
        return w / self.spectral_norm(w)

    def call(self, inputs):
        with tf.keras.backend.name_scope(self.layer.name):
            if not self.layer.built:
                input_shape = tf.keras.backend.int_shape(inputs)
                self.layer.build(input_shape)
                self.layer.built = True
                if self.layer._initial_weights is not None:
                    self.layer.set_weights(self.layer._initial_weights)
        if not hasattr(self.layer, 'spectral_normalization'):
            if hasattr(self.layer, 'kernel'):
                self.layer.kernel = self.spectral_normalization(self.layer.kernel)
            if hasattr(self.layer, 'gamma'):
                self.layer.gamma = self.spectral_normalization(self.layer.gamma)
            self.layer.spectral_normalization = True
        return self.layer(inputs)
class ConvBNleaky_relu(tf.keras.Model):

    def __init__(self, ch, kernelsz=3, strides=1, padding='same'):
        super(ConvBNleaky_relu, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(ch, kernelsz, strides=strides, padding=padding, use_bias=False)
        # self.bn = tf.keras.layers.BatchNormalization()
        self.lrelu = tf.keras.layers.LeakyReLU(.1)


    def call(self, x, training=True):
        return self.lrelu(self.conv1(x))
class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.initializer = tf.keras.initializers.random_normal(0, 0.02)
        self.down = DiscDownsample(16, 4)
        self.down0 = DiscDownsample(32, 4)
        self.down1 = DiscDownsample(64, 4)
        self.down2 = DiscDownsample(128, 4)
        self.down3 = DiscDownsample(128, 4)
        self.down4 = DiscDownsample(256, 4)
        self.down5 = DiscDownsample(256, 4)
        self.down_pred1 = DiscDownsample(32, 4)
        self.down_pred2 = DiscDownsample(128, 4)
        self.down_pred3 = DiscDownsample(128, 4)
        self.down_pred4 = DiscDownsample(256, 4)
        self.down_pred5 = DiscDownsample(128, 4)
        self.conv_out = ConvBNleaky_relu(1, 1)
        self.Flatten = tf.keras.layers.Flatten()
        self.dense_out = tf.keras.layers.Dense(1, use_bias=False, activation="relu")
    def call(self, g_out, real):
        x1_f = self.down(g_out, normal=False)
        x2_f = self.down0(x1_f, normal=False)
        x3_f = self.down1(x2_f, normal=False)
        x4_f = self.down2(x3_f, normal=False)
        x5_f = self.down3(x4_f, normal=False)
        x6_f = self.down4(x5_f, normal=False)

        x1_r = self.down(real, normal=False)
        x2_r = self.down0(x1_r, normal=False)
        x3_r = self.down1(x2_r, normal=False)
        x4_r = self.down2(x3_r, normal=False)
        x5_r = self.down3(x4_r, normal=False)
        x6_r = self.down4(x5_r, normal=False)
        # diff2 = tf.abs(x1_f - x1_r)
        # diff4 = tf.abs(x3_f - x3_r)
        # diff6 = tf.abs(x6_f - x6_r)
        diff2 = tf.abs((x1_f - x1_r)*(tf.nn.softmax(x1_r) + 1))
        diff4 = tf.abs((x3_f - x3_r)*(tf.nn.softmax(x3_r) + 1))
        diff6 = tf.abs((x6_f - x6_r)*(tf.nn.softmax(x6_r) + 1))
        diff2 = self.down_pred2(self.down_pred1(diff2))
        diff4 = tf.concat([diff2, diff4], axis=-1)
        diff4 = self.down_pred5(self.down_pred4(self.down_pred3(diff4)))
        diff6 = tf.concat([diff4, diff6], axis=-1)
        out = self.conv_out(diff6)
        out = self.Flatten(out)
        out = self.dense_out(out)
        return out
if __name__ == '__main__':
    tf.enable_eager_execution()
    a = generator()
    b = a(tf.ones([1, 256, 256, 1]))
    # a.build([1, 256, 256, 1])
    a.summary()