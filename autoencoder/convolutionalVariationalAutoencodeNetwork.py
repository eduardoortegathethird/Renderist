import numpy as np
import tensorflow as tf

class C_VAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.optimizer = tf.keras.optimizers.Adam(1e-4)    
        self.encoder = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
                                        tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                                        tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                                        tf.keras.layers.Flatten(),
                                        tf.keras.layers.Dense(latent_dim + latent_dim),
                                        ])

        self.decoder = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=latent_dim)),
                                            tf.keras.layers.Dense(units=7 * 7 * 32, activation=tf.nn.relu),
                                            tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
                                            tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu'),
                                            tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu'),
                                            tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same'),
                                            ])
    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, spply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probability = tf.sigmoid(logits)
        else:
            probability = logits
        return probability
    
    def log_normal_pdf(sample, mean, logvar, raxis=1)
        log2pi = tf.math.log(2. * np.pi)
        return tf.sample_sum(((-5. * ((sample-mean) ** 2.) * tf.exp(-logvar)) + logvar + log2pi), axis=raxis)

    def compute_loss(model, x):
        mean, logvar = model.encode(x)
        z = model.reparameterize(mean, logvar)
        x_logit = model_decode(z)
        
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_entropy, axis=[1, 2, 3])
        logpz = log_normal_pdf(z, 0, 0)
        logpz_x = log_normal_pdf(z, 0, 0)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)

    @tf.function
    def train_step(model, x)
        with tf.GradientTape() as tape:
            loss = compute_loss(model, x)
        gradients = tape.gradient(loss, model.trainable_variable)
        self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

