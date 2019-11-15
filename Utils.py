import soundfile as sf
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from scipy.io.wavfile import write
from scipy.ndimage import zoom


class noise(object):

    def __init__(self, noise_prefix):

        self.sr = 8000
        self.noise_prefix = noise_prefix
        self.noise_list = []
        file_list = os.listdir(noise_prefix)
        for i in file_list:
            try:
                buffer = sf.read(noise_prefix+i)[0].astype(np.float32)
                buffer = zoom(buffer, 2, order=1)
                self.noise_list.append(buffer)
            except:
                print("Some trouble caused when open noise_files {}".format(i))

    def random_noise(self, max_l):
        idx = np.random.randint(0, len(self.noise_list))
        noise_ = self.noise_list[idx]
        assert len(noise_) > self.sr * 10
        noise_ = np.roll(noise_, np.random.randint(0, len(noise_)))
        noise_ = noise_[100:max_l+100]
        noise_ = noise_ - np.mean(noise_)
        p = np.sum(noise_**2)/len(noise_)
        noise_ = noise_ / np.sqrt(p)
        return noise_, idx

    @staticmethod
    def add_noise_0(datas, min_db=10.,  max_db=10.):
        local_range = np.arange(min_db, max_db+1, 1)
        local_db = np.random.choice(local_range)
        # local_db = np.random.uniform(low=min_db, high=max_db, size=[datas.shape[0]])
        snr = 10 ** (local_db/10)
        power = datas[:, -1, :]**2
        power = np.mean(power, axis=-1)
        n_amp = np.sqrt(power / snr)
        n = np.random.normal(0, 1, size=datas.shape)
        for i in range(2):
            n_amp = np.expand_dims(n_amp, axis=-1)
        labels_stft = datas[:, -1, :]
        noise_map = n * tf.broadcast_to(tf.cast(n_amp, tf.float32), n.shape)
        out = tf.cast(datas, tf.float32) + noise_map
        return out, labels_stft

    def add_noise_1(self, datas, min_db=0.,  max_db=15):
        local_range = np.arange(min_db, max_db+1, 1)
        local_noise = []
        idx = []
        for i in range(datas.shape[0]):
            dts = self.random_noise(datas.shape[2])
            idx .append(dts[1])
            local_noise .append(dts[0])
        local_noise = np.asarray(local_noise)
        local_noise = np.expand_dims(local_noise, axis=1)
        local_db = np.random.choice(local_range)
        # local_db = np.random.uniform(low=min_db, high=max_db, size=[datas.shape[0]])
        snr = 10 ** (local_db/10)
        power = datas[:, -1, :]**2
        power = np.mean(power, axis=-1)
        n_amp = np.sqrt(power / snr)
        n = local_noise
        for i in range(2):
            n_amp = np.expand_dims(n_amp, axis=-1)
        labels_stft = datas[:, -1, :]
        noise_map = n * tf.broadcast_to(tf.cast(n_amp, tf.float32), n.shape)
        out = tf.cast(datas, tf.float32) + noise_map
        return out, labels_stft # , tf.cast(idx, tf.int8)

    def add_noise(self, datas, min_db=7.5,  max_db=7.5):

        i = np.random.uniform(0, 1)
        if i > 0.5:
            out, labels_stft = self.add_noise_1(datas, min_db=min_db,  max_db=max_db)
        else:
            out, labels_stft = self.add_noise_1(datas, min_db=min_db, max_db=max_db)
        return out, labels_stft

class Loss_global(object):

    def __init__(self, LAMBDA= 100):
        self.batch_size = 32
        self.LAMBDA = LAMBDA
        self.canny_x = tf.cast([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], tf.float32)
        self.canny_y = tf.cast([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], tf.float32)
        self.canny_x = tf.expand_dims(self.canny_x, -1)
        self.canny_x = tf.expand_dims(self.canny_x, -1)
        self.canny_y = tf.expand_dims(self.canny_y, -1)
        self.canny_y = tf.expand_dims(self.canny_y, -1)

    @staticmethod
    def gradient_penalty(discriminator, batch_x, fake_image):
        batch_sz = batch_x.shape[0]
        t = tf.random.uniform([batch_sz, 1, 1, 1])
        t = tf.broadcast_to(t, batch_x.shape)
        interplate = t*batch_x + (1 - t) * fake_image
        with tf.GradientTape() as tape:
            tape.watch([interplate])
            d_interplote_logits = discriminator(interplate, False)
        grads = tape.gradient(d_interplote_logits, interplate)
        grads = tf.reshape(grads, [grads.shape[0], -1])
        gp = tf.norm(grads, axis=-1)
        gp = tf.reduce_mean((gp - 1)**2)
        return gp


    def discriminator_loss_w(self, generator, discriminator, recv_sig_sftf, label, tar_dir_local, istraining=True):
        fake_image = generator(recv_sig_sftf, training=istraining, tar_dir=1)
        d_fake_logits = discriminator([fake_image], istraining)
        d_real_logits = discriminator(label, istraining)
        d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_real_logits),
                                                              logits=d_real_logits)
        d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake_logits),
                                                              logits=d_fake_logits)
        ce_loss = tf.reduce_mean(d_loss_real) + tf.reduce_mean(d_loss_fake)
        gp = self.gradient_penalty(discriminator, label, fake_image)
        loss = ce_loss + gp*1.
        return loss, gp, fake_image

    @staticmethod
    def discriminator_loss(disc_real_output, disc_generated_output):
        # real_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(disc_real_output),
        #                                            logits=disc_real_output)
        real_loss = tf.keras.losses.binary_crossentropy(
            tf.ones_like(disc_real_output), disc_real_output, from_logits=True)
        generated_loss = tf.keras.losses.binary_crossentropy(
            tf.zeros_like(disc_generated_output), disc_generated_output, from_logits=True)
        # total_disc_loss = tf.reduce_mean(disc_generated_output - disc_real_output)
        real_loss = tf.reduce_mean(real_loss)
        generated_loss = tf.reduce_mean(generated_loss)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss

    def generator_loss(self, disc_generated_output, gen_output, target, disc_condition=1e3):
        gan_loss = tf.keras.losses.binary_crossentropy(
            tf.ones_like(disc_generated_output), disc_generated_output, from_logits=True)
        # mean absolute error
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output)**2)

        gan_loss = tf.reduce_mean(gan_loss)

        total_gen_loss = gan_loss  + l1_loss*100

        return total_gen_loss

    def dir_loss_func(self, out_sig, label_sig):
        loss = tf.reduce_mean(tf.abs(label_sig - out_sig)*label_sig)
        return loss

    def log_loss_func(self, out_sig, label_sig):
        loss = tf.sqrt(tf.reduce_sum((tf.log1p(out_sig) - tf.log1p(label_sig))**2)/self.batch_size)
        return loss

    def log_loss_func_mean(self, out_sig, label_sig):
        loss = tf.sqrt(tf.reduce_mean((tf.log1p(out_sig) - tf.log1p(label_sig)) ** 2))
        return loss

    def log_loss_sobel(self, out_sig, label_sig, aerfa, threhold):
        audio_x = tf.nn.conv2d(out_sig, filter=self.canny_x, strides=[1, 1, 1, 1], padding="SAME", use_cudnn_on_gpu=True)
        audio_y = tf.nn.conv2d(out_sig, filter=self.canny_y, strides=[1, 1, 1, 1], padding="SAME", use_cudnn_on_gpu=True)
        audio = tf.sqrt(audio_x ** 2 + audio_y ** 2)
        audio = tf.clip_by_value(audio, 0.1 * np.max(audio), 100)
        audio = tf.cast(audio[audio > np.max(audio)*threhold], tf.float32)
        loss = (self.log_loss_func_mean(out_sig, label_sig) + aerfa*np.sum(audio))/(1+aerfa)
        return loss
    def l2_loss(self, out_sig, label_sig):
        return tf.reduce_sum((out_sig - label_sig)**2)/16

def map2one(signals):
    signals = signals
    max = np.max(signals, axis=-1)
    max = np.max(max, axis=-1)
    for i in range(2):
        max = tf.expand_dims(max, axis=-1)
    max = tf.broadcast_to(max, signals.shape)
    signals /= max
    return signals


def Menute_feature(STFTS, ref_mac=-1, z=False, dims=1):
    ref_tfbins = STFTS[:, :, :, ref_mac]/np.max(STFTS[:, :, :, ref_mac])
    STFTS = map2one(STFTS[:, :, :, :ref_mac])
    out_data = tf.expand_dims(tf.cast(ref_tfbins, tf.float32), axis=-1)
    if not z:
        return out_data
    else:
        if dims == 1:
            noise_ = np.random.uniform(0, 0.02, STFTS.shape[0])
            for i in range(3):
                noise_ = tf.expand_dims(noise_, axis=-1)
            noise_ = tf.broadcast_to(noise_, [out_data.shape[0], out_data.shape[1], out_data.shape[2], 1])
        else:
            noise_ = np.random.uniform(0, 0.02, [STFTS.shape[0], STFTS.shape[-1]])
            for i in range(2):
                noise_ = tf.expand_dims(noise_, axis=1)
            noise_ = tf.broadcast_to(noise_, [out_data.shape[0], out_data.shape[1], out_data.shape[2], 1])
        noise_ = tf.cast(noise_, tf.float32)
        return tf.concat([out_data, noise_], axis=-1)

def orthogonal_regularizer(scale) :

    def ortho_reg(w) :
        """ Reshaping the matrxi in to 2D tensor for enforcing orthogonality"""
        _, _, _, c = w.get_shape().as_list()

        w = tf.reshape(w, [-1, c])
        identity = tf.eye(c)

        w_mul = tf.matmul(w, w, transpose_a=True)
        reg = w_mul - identity

        """Calculating the Loss Obtained"""
        ortho_loss = tf.nn.l2_loss(reg)

        return scale * ortho_loss

    return ortho_reg

if __name__ == '__main__':
    noise_local = noise("./noisex-92/")
    for i in range(10):
        buffer = noise_local.random_noise(2**17)[0]
        plt.plot(buffer)
        plt.show()
        write("./haha"+str(i)+".wav", 16000, buffer)
