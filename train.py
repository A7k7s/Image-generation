# 🚀 Improved Conditional GAN for Clearer Image Generation

!pip install tensorflow matplotlib -q

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# ---------------------------
# Parameters
# ---------------------------
embedding_dim = 10
noise_dim = 100
img_size = 32

# ---------------------------
# Generator (Improved)
# ---------------------------
def build_generator():
    noise_input = layers.Input(shape=(noise_dim,))
    text_input = layers.Input(shape=(embedding_dim,))
    
    x = layers.Concatenate()([noise_input, text_input])
    
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(8*8*128, activation='relu')(x)
    x = layers.Reshape((8, 8, 128))(x)
    
    x = layers.Conv2DTranspose(128, (4,4), strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(64, (4,4), strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(1, (3,3), padding='same', activation='tanh')(x)
    
    return tf.keras.Model([noise_input, text_input], x)

# ---------------------------
# Discriminator (Improved)
# ---------------------------
def build_discriminator():
    image_input = layers.Input(shape=(img_size, img_size, 1))
    text_input = layers.Input(shape=(embedding_dim,))
    
    x = layers.Conv2D(64, (3,3), strides=2, padding='same')(image_input)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.Conv2D(128, (3,3), strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.Flatten()(x)
    x = layers.Concatenate()([x, text_input])
    
    x = layers.Dense(128)(x)
    x = layers.LeakyReLU(0.2)(x)
    
    output = layers.Dense(1, activation='sigmoid')(x)
    
    return tf.keras.Model([image_input, text_input], output)

# ---------------------------
# Build Models
# ---------------------------
generator = build_generator()
discriminator = build_discriminator()

discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# ---------------------------
# GAN Model
# ---------------------------
discriminator.trainable = False

noise = layers.Input(shape=(noise_dim,))
text = layers.Input(shape=(embedding_dim,))
generated_image = generator([noise, text])
validity = discriminator([generated_image, text])

gan = tf.keras.Model([noise, text], validity)
gan.compile(loss='binary_crossentropy', optimizer='adam')

# ---------------------------
# Dummy Dataset (32x32)
# ---------------------------
X_train = np.random.normal(0, 1, (1000, img_size, img_size, 1))
text_data = np.random.normal(0, 1, (1000, embedding_dim))

# ---------------------------
# Training
# ---------------------------
epochs = 5000
batch_size = 32

for epoch in range(epochs):
    
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    real_imgs = X_train[idx]
    real_text = text_data[idx]
    
    noise = np.random.normal(0, 1, (batch_size, noise_dim))
    fake_imgs = generator.predict([noise, real_text], verbose=0)
    
    # Train Discriminator
    d_loss_real = discriminator.train_on_batch([real_imgs, real_text], np.ones((batch_size, 1)))
    d_loss_fake = discriminator.train_on_batch([fake_imgs, real_text], np.zeros((batch_size, 1)))
    
    # Train Generator
    noise = np.random.normal(0, 1, (batch_size, noise_dim))
    g_loss = gan.train_on_batch([noise, real_text], np.ones((batch_size, 1)))
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch} | D Loss: {d_loss_real[0]} | G Loss: {g_loss}")

# ---------------------------
# Generate Image
# ---------------------------
noise = np.random.normal(0, 1, (1, noise_dim))
text_sample = np.random.normal(0, 1, (1, embedding_dim))

generated_img = generator.predict([noise, text_sample])

# Normalize for display
img = (generated_img[0] + 1) / 2

plt.imshow(img[:, :, 0], cmap='gray')
plt.title("Clear Generated Image")
plt.axis('off')
plt.show()