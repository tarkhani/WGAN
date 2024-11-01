import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
import matplotlib.pyplot as plt

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Target image size for the dataset and GAN
target_size = (100, 100)
batch_size = 64

# Load and preprocess the dataset
ds = tfds.load('oxford_flowers102', split='train', as_supervised=True)


def preprocess_image(image, label):
    image = tfimage.re.size(image, target_size)
    image = tf.cast(image, tf.float32) / 127.5 - 1.0  # Normalize to [-1, 1]
    return image


ds = ds.map(lambda image, label: preprocess_image(image, label), num_parallel_calls=tf.data.AUTOTUNE)
ds = ds.cache().shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)


# Display a few color samples to verify preprocessing
def display_samples(dataset, num_images=5):
    plt.figure(figsize=(10, 2))
    for i, img in enumerate(dataset.take(num_images)):
        plt.subplot(1, num_images, i + 1)
        plt.imshow((img[0] + 1) / 2)  # Rescale to [0, 1] for display
        plt.axis('off')
    plt.show()


display_samples(ds)


# Generator model for RGB images
def build_generator():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(25 * 25 * 256, input_dim=128),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Reshape((25, 25, 256)),
        tf.keras.layers.UpSampling2D(),
        tf.keras.layers.Conv2D(256, kernel_size=5, padding="same"),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.UpSampling2D(),
        tf.keras.layers.Conv2D(128, kernel_size=5, padding="same"),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Conv2D(64, kernel_size=5, padding="same"),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Conv2D(3, kernel_size=5, padding="same", activation="tanh")  # Output RGB image
    ])
    return model


# Discriminator model for RGB images
def build_discriminator():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, kernel_size=5, strides=2, padding="same", input_shape=(100, 100, 3)),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Conv2D(128, kernel_size=5, strides=2, padding="same"),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Conv2D(256, kernel_size=5, strides=2, padding="same"),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1)  # No activation for WGAN
    ])
    return model


# Wasserstein loss function
def wasserstein_loss(y_true, y_pred):
    return tf.reduce_mean(y_true * y_pred)


# Gradient penalty for WGAN-GP
def gradient_penalty(discriminator, real_images, fake_images):
    batch_size = tf.shape(real_images)[0]
    alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
    interpolated = alpha * real_images + (1 - alpha) * fake_images

    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        pred = discriminator(interpolated)

    grads = tape.gradient(pred, interpolated)
    grads_l2 = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    gradient_penalty = tf.reduce_mean((grads_l2 - 1.0) ** 2)
    return gradient_penalty


# Optimizers
g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005, beta_1=0.5)
d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005, beta_1=0.5)


# Training step for WGAN-GP
@tf.function
def train_step(real_images):
    batch_size = tf.shape(real_images)[0]
    noise = tf.random.normal([batch_size, 128])
    for _ in range(5):  # n_critic = 5 for WGAN-GP
        with tf.GradientTape() as tape:
            fake_images = generator(noise, training=True)
            real_logits = discriminator(real_images, training=True)
            fake_logits = discriminator(fake_images, training=True)
            d_cost = tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)
            gp = gradient_penalty(discriminator, real_images, fake_images)
            d_loss = d_cost + 20.0 * gp  # lambda = 10 for gradient penalty

        d_grads = tape.gradient(d_loss, discriminator.trainable_variables)
        d_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))

    with tf.GradientTape() as tape:
        fake_images = generator(noise, training=True)
        fake_logits = discriminator(fake_images, training=True)
        g_loss = -tf.reduce_mean(fake_logits)

    g_grads = tape.gradient(g_loss, generator.trainable_variables)
    g_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))

    return d_loss, g_loss


# Training function
def train(dataset, epochs):
    for epoch in range(epochs):
        for real_images in dataset:
            d_loss, g_loss = train_step(real_images)

        # Print progress and save sample images periodically
        print(f"Epoch {epoch}, Discriminator Loss: {d_loss.numpy()}, Generator Loss: {g_loss.numpy()}")
        if epoch % 50 == 0:
            save_generated_images(epoch)


# Save generated images
def save_generated_images(epoch, num_images=5):
    test_noise = tf.random.normal([num_images, 128])
    generated_images = generator(test_noise, training=False)
    generated_images = (generated_images + 1) / 2  # Rescale to [0, 1]

    if not os.path.exists("images"):
        os.makedirs("images")

    for i in range(num_images):
        img = tf.image.convert_image_dtype(generated_images[i], dtype=tf.uint8)
        img_path = os.path.join("images", f"generated_img_epoch_{epoch}_{i}.png")
        tf.io.write_file(img_path, tf.image.encode_png(img))
    print(f"Saved generated images for epoch {epoch}.")


# Initialize and train the GAN
generator = build_generator()
discriminator = build_discriminator()
train(ds, epochs=4000)
