import os
import zipfile
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
strategy = tf.distribute.OneDeviceStrategy(device='/gpu:0')

# config = tf.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8)
# # config = tf.ConfigProto()
# sess = tf.Session(config=config)
# sess.run(tf.global_variables_initializer())
# sess.run(tf.local_variables_initializer())

# local_zip = '/tmp/cats_and_dogs_filtered.zip'
# zip_ref = zipfile.ZipFile(local_zip, 'r')
# zip_ref.extractall('/tmp')
# zip_ref.close()

base_dir = '/tmp/pycharm_project_27/DeepLearning/Intern/MURA-v1.1'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'valid')

# # Directory with our training cat pictures
# train_cats_dir = os.path.join(train_dir, 'cats')
#
# # Directory with our training dog pictures
# train_dogs_dir = os.path.join(train_dir, 'dogs')
#
# # Directory with our validation cat pictures
# validation_cats_dir = os.path.join(validation_dir, 'cats')
#
# # Directory with our validation dog pictures
# validation_dogs_dir = os.path.join(validation_dir, 'dogs')

# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(512, activation='relu'),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])

with strategy.scope():
#     model = tf.keras.Sequential([
#     tf.keras.layers.Conv2D(64, kernel_size=(3, 3), input_shape=(224, 224, 3), data_format="channels_last"),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(7, activation=tf.nn.relu),
#     tf.keras.layers.Dense(7, activation=tf.nn.softmax)
# ])

    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(7, activation='sigmoid')
    ])
# Can you guess what the current output shape is at this point? Probably not.
# Let's just print it:

    # model.add(layers.Dense(7,activation=tf.nn.relu))
    # model.add(layers.Dense(7,activation=tf.nn.softmax))
    model.summary()
# The answer was: (40, 40, 32), so we can keep downsampling...



#     model = tf.keras.applications.DenseNet169(weights='imagenet', include_top = False, input_shape=(224, 224, 3))
#     model.trainable = False
#     model = tf.keras.models.Sequential([
#     densenet,
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])
    model.compile(loss=tf.keras.losses.binary_crossentropy,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    train_dir,  # This is the source directory for training images
    target_size=(224, 224),  # All images will be resized to 150x150
    batch_size=20,
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode='categorical')

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=20,
    class_mode='categorical')

history = model.fit(
    train_generator,
    steps_per_epoch=100,  # 2000 images = batch_size * steps
    epochs=100,
    validation_data=validation_generator,
    validation_steps=50,  # 1000 images = batch_size * steps
    verbose=2,workers=1)

