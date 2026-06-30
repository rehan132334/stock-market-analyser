import numpy as np
import tensorflow as tf
from timegan.model import (make_embedder, make_recovery,
                            make_generator, make_supervisor,
                            make_discriminator)


class ConditionalTimeGAN:
    def __init__(self, window_size=30, input_dim=1, hidden_dim=64, noise_dim=32):
        self.window_size = window_size
        self.input_dim   = input_dim
        self.hidden_dim  = hidden_dim
        self.noise_dim   = noise_dim

        # ── Build all 4 components + supervisor ──────────────────────────────
        self.embedder      = make_embedder(window_size, input_dim, hidden_dim)
        self.recovery      = make_recovery(window_size, hidden_dim, input_dim)
        self.generator     = make_generator(window_size, noise_dim, hidden_dim)
        self.supervisor    = make_supervisor(window_size, hidden_dim)
        self.discriminator = make_discriminator(window_size, hidden_dim)

        # ── Separate optimizer per component (Keras 3 requirement) ───────────
        self.opt_er = tf.keras.optimizers.Adam(0.001)   # embedder + recovery
        self.opt_g  = tf.keras.optimizers.Adam(0.001)   # generator only
        self.opt_s  = tf.keras.optimizers.Adam(0.001)   # supervisor only
        self.opt_d  = tf.keras.optimizers.Adam(0.001)   # discriminator only

        self.mse = tf.keras.losses.MeanSquaredError()
        self.bce = tf.keras.losses.BinaryCrossentropy()

    # ── Phase 1: Embedder + Recovery (Autoencoder) ───────────────────────────
    @tf.function
    def _train_autoencoder(self, X_batch, C_batch):
        with tf.GradientTape() as tape:
            H     = self.embedder([X_batch, C_batch], training=True)
            X_hat = self.recovery([H, C_batch], training=True)
            loss  = self.mse(X_batch, X_hat) * 10.0

        grads = tape.gradient(
            loss,
            self.embedder.trainable_variables +
            self.recovery.trainable_variables
        )
        self.opt_er.apply_gradients(zip(
            grads,
            self.embedder.trainable_variables +
            self.recovery.trainable_variables
        ))
        return loss

    # ── Phase 2: Supervisor (latent space temporal dynamics) ─────────────────
    @tf.function
    def _train_supervisor(self, X_batch, C_batch):
        with tf.GradientTape() as tape:
            H    = self.embedder([X_batch, C_batch], training=False)
            S    = self.supervisor(H, training=True)
            loss = self.mse(H[:, 1:, :], S[:, :-1, :])

        grads = tape.gradient(loss, self.supervisor.trainable_variables)
        self.opt_s.apply_gradients(zip(grads, self.supervisor.trainable_variables))
        return loss

    # ── Phase 3a: Generator + Supervisor (joint) ─────────────────────────────
    @tf.function
    def _train_generator(self, X_batch, C_batch):
        batch_size = tf.shape(X_batch)[0]
        noise = tf.random.normal([batch_size, self.window_size, self.noise_dim])

        # persistent=True because we call .gradient() twice from one tape
        with tf.GradientTape(persistent=True) as tape:
            E_hat  = self.generator([noise, C_batch], training=True)
            H_hat  = self.supervisor(E_hat, training=True)
            Y_fake = self.discriminator([H_hat, C_batch], training=False)
            X_hat  = self.recovery([H_hat, C_batch], training=False)

            # Adversarial loss — fool the discriminator
            loss_adv  = self.bce(tf.ones_like(Y_fake), Y_fake)

            # Supervisor loss — maintain temporal dynamics
            loss_sup  = self.mse(E_hat[:, 1:, :], H_hat[:, :-1, :])

            # Moment matching — mean and variance of generated vs real
            loss_mean = tf.reduce_mean(tf.abs(
                tf.reduce_mean(X_hat, axis=0) -
                tf.reduce_mean(X_batch, axis=0)
            ))
            loss_var  = tf.reduce_mean(tf.abs(
                tf.math.reduce_variance(X_hat, axis=0) -
                tf.math.reduce_variance(X_batch, axis=0)
            ))

            # Separate losses for separate optimizers
            loss_g = loss_adv + loss_mean + loss_var
            loss_s = 100.0 * loss_sup

        # Each optimizer only touches its own variables
        grads_g = tape.gradient(loss_g, self.generator.trainable_variables)
        grads_s = tape.gradient(loss_s, self.supervisor.trainable_variables)
        del tape   # must delete persistent tape manually

        self.opt_g.apply_gradients(zip(grads_g, self.generator.trainable_variables))
        self.opt_s.apply_gradients(zip(grads_s, self.supervisor.trainable_variables))

        return loss_g

    # ── Phase 3b: Discriminator ───────────────────────────────────────────────
    @tf.function
    def _train_discriminator(self, X_batch, C_batch):
        batch_size = tf.shape(X_batch)[0]
        noise = tf.random.normal([batch_size, self.window_size, self.noise_dim])

        with tf.GradientTape() as tape:
            # Real path
            H_real = self.embedder([X_batch, C_batch], training=False)
            Y_real = self.discriminator([H_real, C_batch], training=True)

            # Fake path
            E_hat  = self.generator([noise, C_batch], training=False)
            H_fake = self.supervisor(E_hat, training=False)
            Y_fake = self.discriminator([H_fake, C_batch], training=True)

            loss_real = self.bce(tf.ones_like(Y_real),  Y_real)
            loss_fake = self.bce(tf.zeros_like(Y_fake), Y_fake)
            loss      = loss_real + loss_fake

        grads = tape.gradient(loss, self.discriminator.trainable_variables)
        self.opt_d.apply_gradients(zip(grads, self.discriminator.trainable_variables))
        return loss

    # ── Main Training Loop ────────────────────────────────────────────────────
    def train(self, sequences, labels,
              epochs_1=200, epochs_2=200, epochs_3=300,
              batch_size=128):

        dataset = tf.data.Dataset.from_tensor_slices(
            (sequences.astype(np.float32), labels.astype(np.float32))
        ).shuffle(1000).batch(batch_size, drop_remainder=True)

        # ── Phase 1 ───────────────────────────────────────────────────────────
        print("── Phase 1: Autoencoder (Embedder + Recovery) ──")
        for epoch in range(epochs_1):
            for X_b, C_b in dataset:
                loss = self._train_autoencoder(X_b, C_b)
            if epoch % 50 == 0:
                print(f"  Epoch {epoch:>3}/{epochs_1} | recon_loss: {loss:.5f}")
        print(f"  Done. Final recon_loss: {loss:.5f}\n")

        # ── Phase 2 ───────────────────────────────────────────────────────────
        print("── Phase 2: Supervisor ──")
        for epoch in range(epochs_2):
            for X_b, C_b in dataset:
                loss = self._train_supervisor(X_b, C_b)
            if epoch % 50 == 0:
                print(f"  Epoch {epoch:>3}/{epochs_2} | sup_loss: {loss:.5f}")
        print(f"  Done. Final sup_loss: {loss:.5f}\n")

        # ── Phase 3 ───────────────────────────────────────────────────────────
        print("── Phase 3: Joint GAN Training ──")
        for epoch in range(epochs_3):
            g_loss = d_loss = 0.0
            for X_b, C_b in dataset:
                g_loss = self._train_generator(X_b, C_b)

                # Only train discriminator if it's not already too strong
                # This prevents discriminator from dominating early on
                if d_loss < 0.15:
                    pass   # skip — discriminator is strong enough
                else:
                    d_loss = self._train_discriminator(X_b, C_b)

            if epoch % 50 == 0:
                print(f"  Epoch {epoch:>3}/{epochs_3} | g_loss: {g_loss:.5f} | d_loss: {d_loss:.5f}")
        print(f"  Done. Final g_loss: {g_loss:.5f} | d_loss: {d_loss:.5f}\n")

        print("Training complete.")

    # ── Generate Scenarios ────────────────────────────────────────────────────
    def generate(self, condition: np.ndarray, n_samples: int = 100):
        """
        condition : shape (4,) — one-hot [bull, bear, neutral, crash]
        Returns   : shape (n_samples, window_size, 1) — synthetic price sequences
        """
    # Convert both to tensors — no mixing allowed
        C     = tf.constant(np.tile(condition, (n_samples, 1)), dtype=tf.float32)
        noise = tf.random.normal([n_samples, self.window_size, self.noise_dim])

        E_hat = self.generator([noise, C],  training=False)
        H_hat = self.supervisor(E_hat,      training=False)
        X_hat = self.recovery([H_hat, C],   training=False)

        return X_hat.numpy()    # (n_samples, window_size, 1)

    # ── Save Weights ──────────────────────────────────────────────────────────
    def save(self, path="timegan_weights"):
        self.embedder.save_weights(     f"{path}/embedder.weights.h5")
        self.recovery.save_weights(     f"{path}/recovery.weights.h5")
        self.generator.save_weights(    f"{path}/generator.weights.h5")
        self.supervisor.save_weights(   f"{path}/supervisor.weights.h5")
        self.discriminator.save_weights(f"{path}/discriminator.weights.h5")
        print(f"Weights saved to '{path}/'")

    # ── Load Weights ──────────────────────────────────────────────────────────
    def load(self, path="timegan_weights"):
        self.embedder.load_weights(     f"{path}/embedder.weights.h5")
        self.recovery.load_weights(     f"{path}/recovery.weights.h5")
        self.generator.load_weights(    f"{path}/generator.weights.h5")
        self.supervisor.load_weights(   f"{path}/supervisor.weights.h5")
        self.discriminator.load_weights(f"{path}/discriminator.weights.h5")
        print(f"Weights loaded from '{path}/'")