import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# Load CIFAR-100 dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar100.load_data()

# Preprocess the data
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define Inception module
def inception_module(x, filters):
    # 1x1 convolution
    path1 = layers.Conv2D(filters=filters[0], kernel_size=(1, 1), activation='relu', padding='same')(x)

    # 3x3 convolution
    path2 = layers.Conv2D(filters=filters[1], kernel_size=(1, 1), activation='relu', padding='same')(x)
    path2 = layers.Conv2D(filters=filters[2], kernel_size=(3, 3), activation='relu', padding='same')(path2)

    # 5x5 convolution
    path3 = layers.Conv2D(filters=filters[3], kernel_size=(1, 1), activation='relu', padding='same')(x)
    path3 = layers.Conv2D(filters=filters[4], kernel_size=(5, 5), activation='relu', padding='same')(path3)

    # Max pooling
    path4 = layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    path4 = layers.Conv2D(filters=filters[5], kernel_size=(1, 1), activation='relu', padding='same')(path4)

    return layers.concatenate([path1, path2, path3, path4], axis=-1)

# Define GoogLeNet architecture
def googlenet():
    input_layer = layers.Input(shape=(32, 32, 3))

    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(input_layer)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(64, (1, 1), activation='relu')(x)
    x = layers.Conv2D(192, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = inception_module(x, filters=[64, 96, 128, 16, 32, 32])  # Inception 3a
    x = inception_module(x, filters=[128, 128, 192, 32, 96, 64])  # Inception 3b
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = inception_module(x, filters=[192, 96, 208, 16, 48, 64])  # Inception 4a
    x = inception_module(x, filters=[160, 112, 224, 24, 64, 64])  # Inception 4b
    x = inception_module(x, filters=[128, 128, 256, 24, 64, 64])  # Inception 4c
    x = inception_module(x, filters=[112, 144, 288, 32, 64, 64])  # Inception 4d
    x = inception_module(x, filters=[256, 160, 320, 32, 128, 128])  # Inception 4e

    # Additional convolutional layers to maintain spatial dimensions
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = inception_module(x, filters=[256, 160, 320, 32, 128, 128])  # Inception 5a
    x = inception_module(x, filters=[384, 192, 384, 48, 128, 128])  # Inception 5b

    x = layers.AveragePooling2D((1, 1))(x)  # Adjusted pooling size to maintain spatial dimensions
    x = layers.Flatten()(x)
    output_layer = layers.Dense(100, activation='softmax')(x)  # 100 classes for CIFAR-100

    model = models.Model(inputs=input_layer, outputs=output_layer)
    return model

# Instantiate the model
model = googlenet()

# Compile the model
model.compile(optimizer=optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, epochs=100, 
                    validation_data=(test_images, test_labels))

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test accuracy:", test_acc)