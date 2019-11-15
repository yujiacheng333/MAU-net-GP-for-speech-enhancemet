import tensorflow as tf
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
tf.enable_eager_execution()
for j in range(1):
    idx = np.random.randint(0, 2000)

    for i in range(30):
        audio = sf.read("./TIMIT_data/audio_only/"+str(idx)+"_n.wav")[0]
        audio = np.asarray(audio)
        audio = 2*(audio - np.min(audio))/(np.max(audio) - np.min(audio))
        audio -= np.mean(audio)
        pow_a = np.var(audio)
        audio += np.random.normal(0, pow_a*(i+1), size=audio.shape)
        plt.plot(audio)
        plt.show()
        audio = np.expand_dims(audio, axis=0)
        audio = tf.cast(audio, tf.float32)
        audio_s = tf.signal.stft(audio, frame_length=512, frame_step=256,
                                                   fft_length=511,
                                                   pad_end=1)
        audio = tf.expand_dims(tf.abs(audio_s), axis=-1)
        plt.imshow(audio.numpy().squeeze())
        plt.show()
        canny_x = tf.cast([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], tf.float32)
        canny_y = tf.cast([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], tf.float32)
        canny_x = tf.expand_dims(canny_x, -1)
        canny_x = tf.expand_dims(canny_x, -1)
        canny_y = tf.expand_dims(canny_y, -1)
        canny_y = tf.expand_dims(canny_y, -1)
        audio_x = tf.nn.conv2d(audio, filter=canny_x, strides=[1, 1, 1, 1], padding="SAME", use_cudnn_on_gpu=True)
        audio_y = tf.nn.conv2d(audio, filter=canny_y, strides=[1, 1, 1, 1], padding="SAME", use_cudnn_on_gpu=True)
        audio = tf.sqrt(audio_x**2+audio_y**2)
        audio = tf.clip_by_value(audio, 0.1*np.max(audio), 100)
        audio = audio.numpy()
        plt.imshow(np.squeeze(audio))
        plt.show()
        print(tf.linalg.norm(audio, 2).numpy()/tf.linalg.norm(tf.abs(audio_s), 2).numpy())