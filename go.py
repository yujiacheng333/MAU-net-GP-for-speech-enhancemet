import tensorflow as tf
from PIT_loss import noise
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
tf.enable_eager_execution()
a = wavfile.read("input.wav")[1]
a -= tf.reduce_mean(a)
a /= tf.reduce_max(tf.abs(a))

a = tf.cast(a, tf.float32)
a = a[tf.newaxis, tf.newaxis, :]
a = noise(noise_prefix="./noisex-92/").add_noise(a)[0][0]

a = tf.signal.stft(a,
                   frame_step=256,
                   frame_length=512,
                    fft_length=511).numpy()
c = tf.signal.inverse_stft(a,# *tf.exp(1j*tf.cast(a_setas, tf.complex64)),
                            frame_step=256,
                            frame_length=512,
                            fft_length=511)
wavfile.write("./in.wav", 16000, c[0].numpy())
a = tf.cast(a, tf.complex64)
import matplotlib.pyplot as plt
plt.imshow(tf.abs(a)[0])
plt.show()
a_setas = tf.angle(a)
a_ = tf.abs(a)
a_ = a_[..., tf.newaxis]
from main import Ailicenet
AI = Ailicenet()
a_/= tf.reduce_max(a_)
res = AI.gen(a_)
plt.imshow(res[0,:, :, 0])
plt.show()
b=tf.cast(res[...,0], tf.complex64)*a
a = tf.signal.inverse_stft(b,# *tf.exp(1j*tf.cast(a_setas, tf.complex64)),
                            frame_step=256,
                            frame_length=512,
                            fft_length=511)
wavfile.write("./haha_output.wav", 16000, a[0].numpy())

