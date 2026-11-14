# ============================================================
# Day 11 - VAE Image Augmentation
# RetailMind AI
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, Model


# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

data_path = "data/styles.csv"
image_folder = "data/images"

output_folder = "generated_images"

os.makedirs(output_folder, exist_ok=True)


# ------------------------------------------------------------
# 2. LOAD PRODUCT DATA
# ------------------------------------------------------------

df = pd.read_csv(
    data_path,
    on_bad_lines="skip"
)

print("Dataset loaded successfully")

print("\nDataset shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())


# ------------------------------------------------------------
# 3. FIND UNDER-REPRESENTED CATEGORY
# ------------------------------------------------------------
# masterCategory was also used in our earlier
# fashion classification work.

category_counts = df["masterCategory"].value_counts()

print("\nProduct category counts:")
print(category_counts)


# Select the category having the fewest products
rare_category = category_counts.idxmin()

print("\nUnder-represented category:")
print(rare_category)


# ------------------------------------------------------------
# 4. GET IMAGES FROM THE RARE CATEGORY
# ------------------------------------------------------------

rare_data = df[
    df["masterCategory"] == rare_category
].copy()

print(
    "\nNumber of original images:",
    len(rare_data)
)


# ------------------------------------------------------------
# 5. LOAD IMAGES
# ------------------------------------------------------------

image_size = 64

images = []

for product_id in rare_data["id"]:

    image_path = os.path.join(
        image_folder,
        str(product_id) + ".jpg"
    )

    if os.path.exists(image_path):

        try:

            image = tf.keras.utils.load_img(
                image_path,
                target_size=(image_size, image_size)
            )

            image = tf.keras.utils.img_to_array(image)

            # Normalize image between 0 and 1
            image = image / 255.0

            images.append(image)

        except Exception:
            pass


images = np.array(images)

print(
    "Images loaded:",
    images.shape
)


if len(images) == 0:
    raise ValueError(
        "No images were found. Check data/images folder."
    )


# ------------------------------------------------------------
# 6. CREATE VAE ENCODER
# ------------------------------------------------------------

latent_dim = 32


encoder_input = layers.Input(
    shape=(64, 64, 3)
)


x = layers.Conv2D(
    32,
    3,
    strides=2,
    padding="same",
    activation="relu"
)(encoder_input)


x = layers.Conv2D(
    64,
    3,
    strides=2,
    padding="same",
    activation="relu"
)(x)


x = layers.Flatten()(x)


x = layers.Dense(
    128,
    activation="relu"
)(x)


# Mean of latent distribution
z_mean = layers.Dense(
    latent_dim,
    name="z_mean"
)(x)


# Log variance of latent distribution
z_log_var = layers.Dense(
    latent_dim,
    name="z_log_var"
)(x)


# ------------------------------------------------------------
# 7. SAMPLING FUNCTION
# ------------------------------------------------------------

def sampling(args):

    z_mean, z_log_var = args

    epsilon = tf.random.normal(
        shape=tf.shape(z_mean)
    )

    return (
        z_mean
        +
        tf.exp(0.5 * z_log_var)
        *
        epsilon
    )


z = layers.Lambda(
    sampling,
    name="sampling"
)(
    [z_mean, z_log_var]
)


encoder = Model(
    encoder_input,
    [z_mean, z_log_var, z],
    name="encoder"
)


print("\nEncoder:")
encoder.summary()


# ------------------------------------------------------------
# 8. CREATE VAE DECODER
# ------------------------------------------------------------

latent_input = layers.Input(
    shape=(latent_dim,)
)


x = layers.Dense(
    16 * 16 * 64,
    activation="relu"
)(latent_input)


x = layers.Reshape(
    (16, 16, 64)
)(x)


x = layers.Conv2DTranspose(
    64,
    3,
    strides=2,
    padding="same",
    activation="relu"
)(x)


x = layers.Conv2DTranspose(
    32,
    3,
    strides=2,
    padding="same",
    activation="relu"
)(x)


decoder_output = layers.Conv2D(
    3,
    3,
    padding="same",
    activation="sigmoid"
)(x)


decoder = Model(
    latent_input,
    decoder_output,
    name="decoder"
)


print("\nDecoder:")
decoder.summary()


# ------------------------------------------------------------
# 9. CREATE VAE MODEL
# ------------------------------------------------------------

class VAE(Model):

    def __init__(self, encoder, decoder):

        super().__init__()

        self.encoder = encoder
        self.decoder = decoder


    def train_step(self, data):

        # Handle data passed by model.fit()
        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape() as tape:

            z_mean, z_log_var, z = self.encoder(
                data
            )

            reconstruction = self.decoder(z)


            # Reconstruction loss
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(
                        data,
                        reconstruction
                    ),
                    axis=(1, 2)
                )
            )


            # KL divergence loss
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1
                    + z_log_var
                    - tf.square(z_mean)
                    - tf.exp(z_log_var),
                    axis=1
                )
            )


            total_loss = (
                reconstruction_loss
                +
                kl_loss
            )


        gradients = tape.gradient(
            total_loss,
            self.trainable_weights
        )


        self.optimizer.apply_gradients(
            zip(
                gradients,
                self.trainable_weights
            )
        )


        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss
        }


# ------------------------------------------------------------
# 10. TRAIN VAE
# ------------------------------------------------------------

vae = VAE(
    encoder,
    decoder
)


vae.compile(
    optimizer="adam"
)


print("\nTraining VAE...")


vae.fit(
    images,
    epochs=20,
    batch_size=32
)


# ------------------------------------------------------------
# 11. GENERATE SYNTHETIC PRODUCT IMAGES
# ------------------------------------------------------------

number_of_images = 20


random_vectors = tf.random.normal(
    shape=(number_of_images, latent_dim)
)


generated_images = decoder.predict(
    random_vectors
)


print(
    "\nGenerated images:",
    generated_images.shape
)


# ------------------------------------------------------------
# 12. SAVE GENERATED IMAGES
# ------------------------------------------------------------

for i in range(number_of_images):

    image = generated_images[i]

    image = np.clip(
        image * 255,
        0,
        255
    ).astype("uint8")


    output_path = os.path.join(
        output_folder,
        f"synthetic_{i + 1}.png"
    )


    tf.keras.utils.save_img(
        output_path,
        image
    )


print(
    "\nSynthetic images saved in:",
    output_folder
)


# ------------------------------------------------------------
# 13. SHOW GENERATED SAMPLES
# ------------------------------------------------------------
# This allows us to visually inspect whether
# the generated samples look reasonable.

plt.figure(
    figsize=(10, 8)
)


for i in range(
    min(12, number_of_images)
):

    plt.subplot(
        3,
        4,
        i + 1
    )

    plt.imshow(
        generated_images[i]
    )

    plt.axis("off")


plt.suptitle(
    f"Synthetic Images - {rare_category}"
)


plt.tight_layout()


plt.savefig(
    "day11_generated_samples.png"
)


plt.show()


# ------------------------------------------------------------
# 14. SAVE VAE MODELS
# ------------------------------------------------------------

encoder.save(
    "day11_vae_encoder.keras"
)


decoder.save(
    "day11_vae_decoder.keras"
)


print(
    "\nVAE encoder and decoder saved."
)


# ------------------------------------------------------------
# 15. CATEGORY BALANCE RESULT
# ------------------------------------------------------------

original_count = len(rare_data)

augmented_count = (
    original_count
    +
    number_of_images
)


print("\n================================")
print("CATEGORY AUGMENTATION")
print("================================")


print(
    "Category:",
    rare_category
)


print(
    "Original images:",
    original_count
)


print(
    "Synthetic images:",
    number_of_images
)


print(
    "Images after augmentation:",
    augmented_count
)


# ------------------------------------------------------------
# 16. BUSINESS VALUE
# ------------------------------------------------------------

print("\n================================")
print("BUSINESS VALUE")
print("================================")


print(
    "Synthetic product images can increase the number "
    "of training samples available for rare product "
    "categories."
)


print(
    "This can help reduce category imbalance and provide "
    "additional training examples for the product "
    "classification model."
)


print("\nDay 11 completed successfully!")