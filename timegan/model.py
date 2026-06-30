import tensorflow as tf
from tensorflow.keras import layers, Model


def make_embedder(window_size, input_dim, hidden_dim):
    """Encodes real sequences into latent space."""
    seq_input = tf.keras.Input(shape=(window_size, input_dim), name="seq_input")
    cond_input = tf.keras.Input(shape=(4,), name="cond_input")

    # Tile condition across time steps and concatenate
    cond_tiled = layers.RepeatVector(window_size)(cond_input)       # (B, T, 4)
    x = layers.Concatenate(axis=-1)([seq_input, cond_tiled])        # (B, T, input_dim+4)

    x = layers.GRU(hidden_dim, return_sequences=True)(x)
    x = layers.GRU(hidden_dim, return_sequences=True)(x)
    h = layers.Dense(hidden_dim, activation="sigmoid")(x)           # (B, T, H)

    return Model([seq_input, cond_input], h, name="Embedder")


def make_recovery(window_size, hidden_dim, output_dim):
    """Decodes latent space back to sequence space."""
    h_input = tf.keras.Input(shape=(window_size, hidden_dim), name="h_input")
    cond_input = tf.keras.Input(shape=(4,), name="cond_input")

    cond_tiled = layers.RepeatVector(window_size)(cond_input)
    x = layers.Concatenate(axis=-1)([h_input, cond_tiled])

    x = layers.GRU(hidden_dim, return_sequences=True)(x)
    x = layers.GRU(hidden_dim, return_sequences=True)(x)
    x_hat = layers.Dense(output_dim, activation="sigmoid")(x)       # (B, T, 1)

    return Model([h_input, cond_input], x_hat, name="Recovery")


def make_generator(window_size, noise_dim, hidden_dim):
    """Generates synthetic latent vectors from noise + condition."""
    noise_input = tf.keras.Input(shape=(window_size, noise_dim), name="noise_input")
    cond_input = tf.keras.Input(shape=(4,), name="cond_input")

    cond_tiled = layers.RepeatVector(window_size)(cond_input)
    x = layers.Concatenate(axis=-1)([noise_input, cond_tiled])

    x = layers.GRU(hidden_dim, return_sequences=True)(x)
    x = layers.GRU(hidden_dim, return_sequences=True)(x)
    e_hat = layers.Dense(hidden_dim, activation="sigmoid")(x)       # (B, T, H)

    return Model([noise_input, cond_input], e_hat, name="Generator")


def make_supervisor(window_size, hidden_dim):
    """Supervises temporal dynamics in latent space (from paper)."""
    h_input = tf.keras.Input(shape=(window_size, hidden_dim))

    x = layers.GRU(hidden_dim, return_sequences=True)(h_input)
    s = layers.Dense(hidden_dim, activation="sigmoid")(x)

    return Model(h_input, s, name="Supervisor")


def make_discriminator(window_size, hidden_dim):
    """Distinguishes real vs synthetic latent vectors."""
    h_input = tf.keras.Input(shape=(window_size, hidden_dim), name="h_input")
    cond_input = tf.keras.Input(shape=(4,), name="cond_input")

    cond_tiled = layers.RepeatVector(window_size)(cond_input)
    x = layers.Concatenate(axis=-1)([h_input, cond_tiled])

    x = layers.GRU(hidden_dim, return_sequences=True)(x)
    x = layers.GRU(hidden_dim, return_sequences=False)(x)
    y_hat = layers.Dense(1, activation="sigmoid")(x)                # (B, 1)

    return Model([h_input, cond_input], y_hat, name="Discriminator")