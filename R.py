import tensorflow as tf
import numpy as np
from tensorflow.python.keras import layers

class ConvBNRelu(tf.keras.Model):

    def __init__(self, ch, kernelsz=3, strides=1, padding='same'):
        super(ConvBNRelu, self).__init__()

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(ch, kernelsz, strides=strides, padding=padding),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])

    def call(self, x, training=True):
        x = self.model(x, training=training)
        return x

def upconv3x3(channels, stride=1, kernel=(3, 3)):

    return layers.Conv2DTranspose(filters=channels,
                                  kernel_size=kernel,
                                  strides=stride,
                                  padding='same',
                                  use_bias=False,
                                  kernel_initializer=tf.random_normal_initializer())

class InverseResnetBlock(tf.keras.Model):

    def __init__(self, channels, strides, residual_path=False):
        super(InverseResnetBlock, self).__init__()

        self.channels = channels
        self.strides = strides
        self.residual_path = residual_path
        self.conv1 = upconv3x3(channels, strides)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = upconv3x3(channels)
        self.bn2 = layers.BatchNormalization()

        if residual_path:
            self.up_conv = upconv3x3(channels, stride=strides)
            self.upbn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=True, mask=None):
        residual = inputs
        x = self.bn1(inputs, training=training)
        x = tf.nn.relu(x)
        x = self.conv1(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        if self.residual_path:
            residual = self.upbn(inputs, training=training)
            residual = tf.nn.relu(residual)
            residual = self.up_conv(residual)
        x = x + residual
        return x

class Upsample(tf.keras.Model):
    def __init__(self, filters, kernel_size, apply_dropout=False):
        super(Upsample, self).__init__()
        self.apply_dropout = apply_dropout
        self.initializer = tf.keras.initializers.random_normal(0, 0.02)
        self.up_conv = tf.keras.layers.Conv2DTranspose(filters=filters,
                                                       kernel_size=kernel_size,
                                                       strides=2,
                                                       padding="same",
                                                       kernel_initializer=self.initializer,
                                                       use_bias=False)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        if self.apply_dropout:
            self.dropout = tf.keras.layers.Dropout(0.5)

    def call(self, x1, x2, training):
        x = self.up_conv(x1)
        x = self.batch_norm(x, training=training)
        if self.apply_dropout:
            x = self.dropout(x, training=training)
        x = tf.nn.relu6(x)
        x = tf.concat([x, x2], axis=-1)
        return x

class Separate2block(tf.keras.Model):
    def __init__(self, block_count=16):
        super(Separate2block, self).__init__()
        self.block_count = block_count
    def call(self, inputs, training=True):
        bs = inputs.shape[0]
        time = inputs.shape[1]
        org_chs = inputs.shape[-1]
        sz = int(time.value/np.sqrt(self.block_count))
        return tf.reshape(inputs, [bs, sz, sz, self.block_count*org_chs])

def conv3x3(channels, stride=1, kernel=(3, 3)):

    return layers.Conv2D(filters=channels,
                         kernel_size=kernel,
                         strides=stride,
                         padding='same',
                         use_bias=False,
                         kernel_initializer=tf.random_normal_initializer())

class block_wise_attention_tfwise(tf.keras.Model):

    def __init__(self, chs, mid_attention_rate=1/2, block_counts=16, tfwise=False):
        super(block_wise_attention_tfwise, self).__init__()
        self.use_tfwise = tfwise
        self.block_counts = block_counts
        self.sep = Separate2block(block_count=self.block_counts)
        self.mid_attention_rate = mid_attention_rate
        self.conv1 = ConvBNRelu(int(chs / 2 *self.block_counts), kernelsz=2, strides=1)
        self.conv2 = ConvBNRelu(int(chs*self.block_counts), kernelsz=2, strides=1)
        self.dense1 = layers.Dense(int(self.mid_attention_rate * chs))
        self.dense2 = layers.Dense(chs)
    def tfwise_attention(self, blocks, training=True): # bs 64 64 16
        blocks = self.conv1(blocks, training)
        blocks = self.conv2(blocks, training)
        return blocks
    def call(self, inputs, training=True, mask=None):
        blocks = self.sep(inputs)
        if self.use_tfwise:
            b_attention = self.tfwise_attention(blocks, training)
        avgs = layers.GlobalAveragePooling2D()(blocks)
        attention = self.dense1(avgs)
        attention = self.dense2(attention)
        attention = tf.nn.sigmoid(attention)
        for i in range(2):
            attention = tf.expand_dims(attention, axis=1)
        attention = tf.broadcast_to(attention, blocks.shape)
        if self.use_tfwise:
            attention *= b_attention
        attention = tf.reshape(attention, inputs.shape)
        return attention

class trunck(tf.keras.Model):
    def __init__(self, chs, block_counts, mid_attention_rate):
        super(trunck, self).__init__()
        self.conv1 = ConvBNRelu(chs, strides=1)
        self.attention = block_wise_attention_tfwise(chs=chs,
                                                     block_counts=block_counts,
                                                     mid_attention_rate=mid_attention_rate)
        self.conv2 = ConvBNRelu(chs, strides=2)
    def call(self, inputs, training=True):
        x = self.conv1(inputs, training)
        attention = self.attention(x)*x
        return self.conv2(x+attention, training), x

class Encoder(tf.keras.Model):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = layers.Conv2D(16, kernel_size=7, padding="same", dilation_rate=(2, 2))
        self.bn = layers.BatchNormalization()
        self.conv2 = trunck(32, 16, 1/2)
        self.conv3 = trunck(64, 16, 1/4)
        self.conv4 = trunck(128, 16, 1/4)
        self.conv5 = trunck(256, 16, 1/8)
        self.conv6 = trunck(512, 16, 1/8)

    def call(self, inputs, training=True):
        x = self.conv1(inputs)
        x = self.bn(x)
        x, x1 = self.conv2(x, training)
        x, x2 = self.conv3(x, training)
        x, x3 = self.conv4(x, training)
        x, x4 = self.conv5(x, training)
        x, x5 = self.conv6(x, training)
        return x, x5, x4, x3, x2, x1



class Decoder(tf.keras.Model):
    def __init__(self):
        super(Decoder, self).__init__()
        self.initializer = tf.keras.initializers.RandomNormal(0, 0.02)
        self.up1 = Upsample(512, 4, True)
        self.up2 = Upsample(256, 4, True)
        self.up3 = Upsample(256, 4, True)
        self.up4 = Upsample(128, 4, True)
        self.last = tf.keras.layers.Conv2DTranspose(filters=1,
                                                    kernel_size=(4, 4),
                                                    strides=2,
                                                    padding='same',
                                                    kernel_initializer=self.initializer)

    def call(self, input, training=True):
        x5, x4, x3, x2, x1 = input
        x = self.up1(x, x7, training=training)
        x = self.up2(x, x6, training=training)
        x = self.up3(x, x4, training=training)
        x = self.up4(x, x3, training=training)
        x = self.last(x)
        x = tf.nn.relu(x)
        return x

class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    def call(self, inputs, training=True):
        x = self.encoder(inputs, training)
        x = self.decoder(x, training)
        return x
if __name__ == '__main__':
    test_input = tf.cast(np.ones([32, 256, 256, 1]), tf.float32)
    a = Encoder()
    out = a(test_input)
    print(out.shape)