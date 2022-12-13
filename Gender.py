from datetime import datetime

import tensorflow as tf
from keras import layers, models, optimizers, losses

# Shapes:
batch_size = 20
img_height = 200
img_width = 200

# Import Data:
ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    'GenderDataset/Training',
    label_mode="int",
    class_names=['female', 'male'],
    batch_size=batch_size,
    image_size=(img_height, img_width)  # reshapes bc different heights and width
)
ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
    'GenderDataset/Validation',
    label_mode="int",
    class_names=['female', 'male'],
    batch_size=batch_size,
    image_size=(img_height, img_width)  # reshapes bc different heights and width
)

# Build model
model = models.Sequential()
# Input Layer
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
model.add(layers.MaxPooling2D((2, 2)))
# Convolution und Pooling Layers
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
# Prädikations Layer
model.add(layers.Dense(512, activation='relu'))
# Output
model.add(layers.Dense(1, activation='sigmoid'))
# Compile
model.compile(loss=losses.binary_crossentropy, optimizer=optimizers.Adadelta(), metrics=['acc'])
print(model.summary())

# Tensorboard
log_dir = "logs/Gender/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Training the Model
model.fit(
    ds_train,
    epochs=30,
    validation_data=ds_validation,
    callbacks=[tensorboard_callback]
)

# Save Model
model.save('Gender_2.h5')
