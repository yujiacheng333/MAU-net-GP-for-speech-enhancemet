import tensorflow as tf
# from backup import ops2
import ops as ops2
import Utils
import get_tfrecord_n
import os
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.backend import max as bmax

"""
Be Iris for you
surpass CGANS frame work named gradient provider + G
"""

class Ailicenet(object):

    def __init__(self, use_bigru=False, lr=1e-4, use_w=False):

        super(Ailicenet, self).__init__()
        self.disc_condition = 1e3
        self.use_w = use_w
        self.checkpoint_dir = './training_checkpoints/gan'
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.use_gan_dir = os.path.join(self.checkpoint_dir, "gan")
        os.makedirs(self.use_gan_dir, exist_ok=True)
        self.helper = get_tfrecord_n.get_tfrecord()
        self.lr = lr
        self.gen = ops2.generator()
        self.disc = ops2.Discriminator()
        self.gen_optimizer = tf.train.RMSPropOptimizer(self.lr)
        self.disc_optimizer = tf.train.RMSPropOptimizer(self.lr)
        self.checkpoint = tf.train.Checkpoint(gen_optimizer=self.gen_optimizer,
                                              gen=self.gen,
                                              dicrim_optimizer=self.disc_optimizer,
                                              disc=self.disc)
        self.ckpt_manager = tf.contrib.checkpoint.CheckpointManager(
            self.checkpoint,
            directory=self.checkpoint_dir,
            max_to_keep=5
        )
        self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
        self.epo = self.helper.epo
        self.Iter = self.helper.Iter
        self.Iter_test = self.helper.Iter_test
        assert self.Iter is not None
        self.loss = Utils.Loss_global()
        self.train_step = 0

    def make_stft(self, recv_sig_sftf, label_signal_stft):

        recv_sig_sftf = tf.abs(tf.signal.stft(recv_sig_sftf,
                                              frame_length=512,
                                              frame_step=256,
                                              fft_length=511,
                                              pad_end=1))

        label_signal_stft = tf.signal.stft(label_signal_stft,
                                           frame_length=512,
                                           frame_step=256,
                                           fft_length=511,
                                           pad_end=1)
        seta = tf.angle(label_signal_stft)

        label_signal_stft = tf.abs(label_signal_stft)
        return recv_sig_sftf, label_signal_stft, seta

    def train_local(self):
        train_step = 0
        noise_method = Utils.noise("./noisex-92/")
        # try:
        if 1:
            while 1:
                with tf.device("/CPU:0"):
                    [recv_sig_sftf, _, _] = self.Iter.get_next()  # __ == labels_signal with noise and _ means inputseta
                    train_step += 1
                bs = recv_sig_sftf.shape[0]
                chs = recv_sig_sftf.shape[1]
                L = recv_sig_sftf.shape[-1]
                recv_sig_sftf, label_signal_stft = noise_method.add_noise(datas=recv_sig_sftf)
                recv_sig_sftf = tf.reshape(recv_sig_sftf, [-1, L])
                recv_sig_sftf, label_signal_stft, seta = self.make_stft(recv_sig_sftf, label_signal_stft)
                recv_sig_sftf = tf.transpose(tf.reshape(recv_sig_sftf, [bs, chs, 256, 256]), [0, 2, 3, 1])
                noise_map = (label_signal_stft - recv_sig_sftf[:, :, :, -1])**2
                label_map = label_signal_stft**2
                label_mask = tf.sqrt(label_map/(label_map + noise_map + 1e-5))[... ,tf.newaxis].numpy()
                label_mask = tf.clip_by_value(label_mask, 0, 1)
                with tf.device("/CPU:0"):
                    recv_sig_sftf = tf.expand_dims(recv_sig_sftf[..., -1], axis=-1)

                    amp = bmax(recv_sig_sftf, axis=[1, 2, 3])
                    recv_sig_sftf /= tf.broadcast_to(amp[:, tf.newaxis, tf.newaxis, tf.newaxis], recv_sig_sftf.shape)
                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                    gen_out_data = self.gen(recv_sig_sftf, training=True)
                    disc_fake = self.disc(gen_out_data, label_mask)
                    if train_step % 1000 == 0:
                      plt.imshow(gen_out_data[0, :, :, 0])
                      plt.show()
                        # print(self.disc(label_mask, label_mask, noise_idx))
                    t = tf.random.uniform(minval=0, maxval=.5, shape=[16, 1, 1, 1])
                    t_ = tf.broadcast_to(t, gen_out_data.shape)

                    interplate = t_ * gen_out_data + (1 - t_) * label_mask  # t_ bigger not same
                    disc_real = self.disc(interplate, label_mask)
                    disc_loss = tf.reduce_mean(tf.nn.relu(1 - disc_fake)) + tf.reduce_mean(
                        tf.abs(tf.squeeze(t) - disc_real))
                    disc_grads = disc_tape.gradient(disc_loss, self.disc.trainable_variables)

                    self.disc_optimizer.apply_gradients(zip(disc_grads, self.disc.trainable_variables))
                    # gen_loss = tf.abs(tf.reduce_mean(disc_fake))
                    # gen_loss = tf.abs(tf.reduce_mean(disc_fake - disc_real))
                    log_loss = self.loss.log_loss_func(gen_out_data, label_mask)
                    #if log_loss > 80:
                        # continue
                    l1_loss = tf.reduce_mean(tf.abs(gen_out_data - label_mask))
                    l2_loss = self.loss.l2_loss(gen_out_data, label_mask)
                    gen_loss = tf.nn.relu(1-disc_loss)*tf.reduce_mean(disc_fake) + tf.nn.relu(l2_loss - 1100) * 10 + l2_loss*1e-4
                    # gen_loss = tf.reduce_mean(disc_fake)
                    gen_gradients = gen_tape.gradient(gen_loss,
                                                      self.gen.trainable_variables)

                    self.gen_optimizer.apply_gradients(zip(gen_gradients, self.gen.trainable_variables))
                gen_out_data = gen_out_data.numpy()
                if train_step % 100 == 0:

                    gen_out_data = gen_out_data * tf.expand_dims(recv_sig_sftf[:, :, :, -1], axis=-1)
                    stfts = tf.cast(gen_out_data.numpy()[:, :, :, -1], tf.complex64)
                    stfts = tf.cast(stfts.numpy() * np.exp(1j * seta.numpy()), tf.complex64)
                    audios = tf.signal.inverse_stft(stfts,
                                                    frame_step=256,
                                                    frame_length=512,
                                                    fft_length=511)

                    for ct, au in enumerate(audios):
                        au = au.numpy()
                        # au = 2 * (au - np.min(au)) / (np.max(au) - np.min(au)) - 1
                        au = au / np.max(np.abs(au))
                        wavfile.write("./result/gan/gen/haha_gen" + str(ct) + ".wav", 16000, au)
                    stfts = tf.cast(recv_sig_sftf.numpy()[:, :, :, -1], tf.complex64)
                    stfts = tf.cast(stfts.numpy() * np.exp(1j * seta.numpy()), tf.complex64)
                    audios = tf.signal.inverse_stft(stfts,
                                                    frame_step=256,
                                                    frame_length=512,
                                                    fft_length=511)

                    for ct, au in enumerate(audios):
                        au = au.numpy()
                        # au = 2 * (au - np.min(au)) / (np.max(au) - np.min(au)) - 1
                        au = au / np.max(np.abs(au))
                        wavfile.write("./result/gan/rec/haha_rec" + str(ct) + ".wav", 16000, au)

                    stfts = tf.cast(label_signal_stft.numpy()[:, :, :], tf.complex64)
                    stfts = tf.cast(stfts.numpy() * np.exp(1j * seta.numpy()), tf.complex64)
                    audios = tf.signal.inverse_stft(stfts,
                                                    frame_step=256,
                                                    frame_length=512,
                                                    fft_length=511)

                    for ct, au in enumerate(audios):
                        au = au.numpy()
                        # au = 2 * (au - np.min(au)) / (np.max(au) - np.min(au)) - 1
                        au = au / np.max(np.abs(au))
                        wavfile.write("./result/gan/la/haha_la" + str(ct) + ".wav", 16000, au)
                if train_step % 20 == 0:
                    self.ckpt_manager.save()

                # if gp > 10:
                  #   break
                print("after {} iter training the gen loss is {} disc_loss is {} ,  L1 loss is _{}, l2 {}".format(train_step, gen_loss.numpy(), disc_loss.numpy(), l1_loss, l2_loss))
                with open("haha.txt", "a+") as f:
                    f.write(str(l1_loss.numpy()) + ",")
    def test(self):
        train_step = 0
        noise_method = Utils.noise("./noisex-92/")
        # try:
        iter = 0
        if 1:
            for i in range(10):
                [recv_sig_sftf, _, _] = self.Iter_test.get_next()  # __ == labels_signal with noise and _ means inputseta
                bs = recv_sig_sftf.shape[0]
                chs = recv_sig_sftf.shape[1]
                L = recv_sig_sftf.shape[-1]
                recv_sig_sftf, label_signal_stft = noise_method.add_noise(datas=recv_sig_sftf)
                recv_sig_sftf = tf.reshape(recv_sig_sftf, [-1, L])
                recv_sig_sftf, label_signal_stft, seta = self.make_stft(recv_sig_sftf, label_signal_stft)
                recv_sig_sftf = recv_sig_sftf.numpy()
                recv_sig_sftf[recv_sig_sftf < 1e-9] = -1
                recv_sig_sftf = tf.transpose(tf.reshape(recv_sig_sftf, [bs, chs, 256, 256]), [0, 2, 3, 1])
                label_mask = tf.expand_dims(
                    tf.cast(label_signal_stft.numpy() / recv_sig_sftf.numpy()[:, :, :, -1], tf.float32),
                    axis=-1).numpy()
                label_mask[label_mask < 0] = 0
                recv_sig_sftf = Utils.Menute_feature(recv_sig_sftf, z=False).numpy()
                label_signal_stft = tf.expand_dims(label_signal_stft, axis=-1).numpy()
                recv_sig_sftf = tf.expand_dims(tf.cast(recv_sig_sftf, tf.float32)[:, :, :, -1], axis=-1)
                gen_out_data = self.gen(recv_sig_sftf, training=False)
                iter += 1
                log_loss = self.loss.log_loss_func(gen_out_data, label_mask)
                l1_loss = tf.reduce_mean(tf.abs(gen_out_data - label_mask))
                gen_out_data = gen_out_data.numpy()
                plt.imshow(gen_out_data[0, :, :, 0])
                plt.show()
                gen_out_data = gen_out_data * tf.expand_dims(recv_sig_sftf[:, :, :, -1], axis=-1)
                stfts = tf.cast(gen_out_data.numpy()[:, :, :, -1], tf.complex64)
                stfts = tf.cast(stfts.numpy() * np.exp(1j * seta.numpy()), tf.complex64)
                audios = tf.signal.inverse_stft(stfts,
                                                frame_step=256,
                                                frame_length=512,
                                                fft_length=511)

                for ct, au in enumerate(audios):
                    au = au.numpy()
                    # au = 2 * (au - np.min(au)) / (np.max(au) - np.min(au)) - 1
                    au = au / np.max(np.abs(au))
                    wavfile.write("./result/gan/gen/haha_{}gen".format(iter) + str(ct) + ".wav", 16000, au)
                stfts = tf.cast(recv_sig_sftf.numpy()[:, :, :, -1], tf.complex64)
                stfts = tf.cast(stfts.numpy() * np.exp(1j * seta.numpy()), tf.complex64)
                audios = tf.signal.inverse_stft(stfts,
                                                frame_step=256,
                                                frame_length=512,
                                                fft_length=511)

                for ct, au in enumerate(audios):
                    au = au.numpy()
                    # au = 2 * (au - np.min(au)) / (np.max(au) - np.min(au)) - 1
                    au = au / np.max(np.abs(au))
                    wavfile.write("./result/gan/rec/haha_{}rec".format(iter) + str(ct) + ".wav", 16000, au)

                stfts = tf.cast(label_signal_stft[:, :, :, -1], tf.complex64)
                stfts = tf.cast(stfts.numpy() * np.exp(1j * seta.numpy()), tf.complex64)
                audios = tf.signal.inverse_stft(stfts,
                                                frame_step=256,
                                                frame_length=512,
                                                fft_length=511)

                for ct, au in enumerate(audios):
                    au = au.numpy()
                    # au = 2 * (au - np.min(au)) / (np.max(au) - np.min(au)) - 1
                    au = au / np.max(np.abs(au))
                    wavfile.write("./result/gan/la/haha_{}la".format(iter) + str(ct) + ".wav", 16000, au)
                if train_step % 20 == 0:
                    self.ckpt_manager.save()

                # if gp > 10:
                #   break
                print("L1 loss is _{}, log_loss is {}".format(self.loss.dir_loss_func(gen_out_data, label_mask), l1_loss))
                with open("haha.txt", "a+") as f:
                    f.write(str(l1_loss.numpy()) + ",")

if __name__ == '__main__':
    ai = Ailicenet()
    ai.train_local()
    # ai.test()
