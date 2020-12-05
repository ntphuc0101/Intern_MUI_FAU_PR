
import os
import zipfile
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# tf.logging.set_verbosity(tf.logging.ERROR)
devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devices[0], True)

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import numpy as np
import os
# import matplotlib.pyplot as plt
from matplotlib import pyplot as plt

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
strategy = tf.distribute.OneDeviceStrategy(device='/gpu:0')


# Class Imbalance for only
base_dir = '/tmp/pycharm_project_27/DeepLearning/Intern/tmp/MURA-v1.1'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'valid')

labels = ['XR_ELBOW','XR_FINGER','XR_FOREARM','XR_HAND','XR_HUMERUS','XR_SHOULDER','XR_WRIST']

print("Train set:\n========================================")
num_XR_ELBOW = len(os.listdir(os.path.join(train_dir, 'XR_ELBOW')))
num_XR_FINGER = len(os.listdir(os.path.join(train_dir, 'XR_FINGER')))
num_XR_FOREARM = len(os.listdir(os.path.join(train_dir, 'XR_FOREARM')))
num_XR_HAND = len(os.listdir(os.path.join(train_dir, 'XR_HAND')))
num_XR_HUMERUS = len(os.listdir(os.path.join(train_dir, 'XR_HUMERUS')))
num_XR_SHOULDER = len(os.listdir(os.path.join(train_dir, 'XR_SHOULDER')))
num_XR_WRIST = len(os.listdir(os.path.join(train_dir, 'XR_WRIST')))
print(f"XR_ELBOW={num_XR_ELBOW}")
print(f"XR_XR_FINGER={num_XR_FINGER}")
print(f"XR_FOREARM={num_XR_FOREARM}")
print(f"XR_XR_HAND={num_XR_HAND}")
print(f"XR_HUMERUS={num_XR_HUMERUS}")
print(f"XR_XR_SHOULDER={num_XR_SHOULDER}")
print(f"XR_XR_WRIST={num_XR_WRIST}")

print("Valid set:\n========================================")
print(f"XR_ELBOW={len(os.listdir(os.path.join(validation_dir, 'XR_ELBOW')))}")
print(f"XR_XR_FINGER={len(os.listdir(os.path.join(validation_dir, 'XR_FINGER')))}")
print(f"XR_FOREARM={len(os.listdir(os.path.join(validation_dir, 'XR_FOREARM')))}")
print(f"XR_XR_HAND={len(os.listdir(os.path.join(validation_dir, 'XR_HAND')))}")
print(f"XR_HUMERUS={len(os.listdir(os.path.join(validation_dir, 'XR_HUMERUS')))}")
print(f"XR_XR_SHOULDER={len(os.listdir(os.path.join(validation_dir, 'XR_SHOULDER')))}")
print(f"XR_XR_WRIST={len(os.listdir(os.path.join(validation_dir, 'XR_WRIST')))}")


weight_for_num_XR_ELBOW = num_XR_ELBOW / (num_XR_ELBOW + num_XR_FINGER + num_XR_FOREARM + num_XR_HAND + num_XR_HUMERUS + num_XR_SHOULDER + num_XR_ELBOW)
weight_for_num_XR_FINGER = num_XR_FINGER / (num_XR_ELBOW + num_XR_FINGER + num_XR_FOREARM + num_XR_HAND + num_XR_HUMERUS + num_XR_SHOULDER + num_XR_ELBOW)
weight_for_num_XR_FOREARM= num_XR_FOREARM / (num_XR_ELBOW + num_XR_FINGER + num_XR_FOREARM + num_XR_HAND + num_XR_HUMERUS + num_XR_SHOULDER + num_XR_ELBOW)
weight_for_num_XR_HAND = num_XR_HAND / (num_XR_ELBOW + num_XR_FINGER + num_XR_FOREARM + num_XR_HAND + num_XR_HUMERUS + num_XR_SHOULDER + num_XR_ELBOW)
weight_for_num_XR_HUMERUS = num_XR_HUMERUS / (num_XR_ELBOW + num_XR_FINGER + num_XR_FOREARM + num_XR_HAND + num_XR_HUMERUS + num_XR_SHOULDER + num_XR_ELBOW)
weight_for_num_XR_SHOULDER = num_XR_SHOULDER / (num_XR_ELBOW + num_XR_FINGER + num_XR_FOREARM + num_XR_HAND + num_XR_HUMERUS + num_XR_SHOULDER + num_XR_ELBOW)
weight_for_num_XR_WRIST = num_XR_WRIST / (num_XR_ELBOW + num_XR_FINGER + num_XR_FOREARM + num_XR_HAND + num_XR_HUMERUS + num_XR_SHOULDER + num_XR_ELBOW)



class_weight = {0: weight_for_num_XR_ELBOW,
                1: weight_for_num_XR_FINGER,
                2: weight_for_num_XR_FOREARM,
                3: weight_for_num_XR_HAND,
                4:weight_for_num_XR_HUMERUS,
                5: weight_for_num_XR_SHOULDER,
                6: weight_for_num_XR_WRIST,
                }

print(f"Weight for class weight_for_num_XR_ELBOW: {weight_for_num_XR_ELBOW:.2f}")
print(f"Weight for class weight_for_num_XR_FINGER: {weight_for_num_XR_FINGER:.2f}")
print(f"Weight for class weight_for_num_XR_FOREARM: {weight_for_num_XR_FOREARM:.2f}")
print(f"Weight for class weight_for_num_XR_HAND: {weight_for_num_XR_HAND:.2f}")
print(f"Weight for class weight_for_num_XR_HUMERUS: {weight_for_num_XR_HUMERUS:.2f}")
print(f"Weight for class weight_for_num_XR_SHOULDER: {weight_for_num_XR_SHOULDER:.2f}")
print(f"Weight for class weight_for_num_XR_WRIST: {weight_for_num_XR_WRIST:.2f}")

#normalization
with strategy.scope():
    def H(inputs, num_filters, dropout_rate):

        x = tf.keras.layers.BatchNormalization(epsilon=eps)(inputs)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.ZeroPadding2D((1, 1))(x)
        x = tf.keras.layers.Conv2D(num_filters, kernel_size=(3, 3), use_bias=False, kernel_initializer='he_normal')(x)
        x = tf.keras.layers.Dropout(rate=dropout_rate)(x)
        return x


    def transition(inputs, num_filters, compression_factor, dropout_rate):
        # compression_factor is the 'θ'
        x = tf.keras.layers.BatchNormalization(epsilon=eps)(inputs)
        x = tf.keras.layers.Activation('relu')(x)
        num_feature_maps = inputs.shape[1]  # The value of 'm'

        x = tf.keras.layers.Conv2D(np.floor(compression_factor * num_feature_maps).astype(np.int),
                                   kernel_size=(1, 1), use_bias=False, padding='same', kernel_initializer='he_normal',
                                   kernel_regularizer=tf.keras.regularizers.l2(0))(x)
        x = tf.keras.layers.Dropout(rate=dropout_rate)(x)

        x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
        return x


    def dense_block(inputs, num_layers, num_filters, growth_rate, dropout_rate):
        for i in range(num_layers):  # num_layers is the value of 'l'
            conv_outputs = H(inputs, num_filters, dropout_rate)
            inputs = tf.keras.layers.Concatenate()([conv_outputs, inputs])
            num_filters += growth_rate  # To increase the number of filters for each layer.
        return inputs, num_filters


    input_shape = (224, 224, 3)
    num_blocks = 1
    num_layers_per_block = 1
    growth_rate = 16
    dropout_rate = 0
    compress_factor = 0.5
    eps = 1.1e-5

    num_filters = 16

    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=(3, 3), use_bias=False, kernel_initializer='he_normal',
                               kernel_regularizer=tf.keras.regularizers.l2(0))(inputs)

    for i in range(num_blocks):
        x, num_filters = dense_block(x, num_layers_per_block, num_filters, growth_rate, dropout_rate)
        x = transition(x, num_filters, compress_factor, dropout_rate)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(7)(x)  # Num Classes for CIFAR-10
    outputs = tf.keras.layers.Activation('softmax')(x)
    model = tf.keras.models.Model(inputs, outputs)
    model.summary()
    # SparseCategoricalCrossentropy(from_logits=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',patience=3, mode='min')
    model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(lr=0.001),
              metrics=['accuracy',tf.keras.metrics.AUC(),tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])


# All images will be rescaled by 1./255
# data during training by applying random lateral inversions and
# rotations of up to 30 degree
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    # rotation_range=10,
    # zoom_range=0.1,
    # horizontal_flip=True,
    samplewise_center = True,
    samplewise_std_normalization=True
)
#

test_datagen = ImageDataGenerator(rescale=1. / 255,
                  # horizontal_flip = False,
                  # zoom_range = 0.0,
                  samplewise_std_normalization=True,
                  samplewise_center = True
                                  )
batch_size = 2
# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    train_dir,  # This is the source directory for training images
    target_size=(224, 224),  # All images will be resized to 150x150
    batch_size=batch_size,
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode='categorical')
print("train_generator", train_generator.total_batches_seen)
# Flow validation images in batches of 20 using test_datagen generator
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical')
print("validation_generator",validation_generator.total_batches_seen)


def plotImages(images_arr):
    plt.figure()
    fig, axes = plt.subplots(1,6, figsize=(40,40))

    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        # plt.imsave('/tmp/pycharm_project_27/DeepLearning/Intern/AugumenationTra.png',img)
    plt.tight_layout()
    plt.show()
    plt.savefig('/tmp/pycharm_project_27/DeepLearning/Intern/AugumenationTra.png', dpi=1000)

# import numpy as np
# def plots(ims, figsize=(40,40), rows=1, interp=False, titles=None):
#     if type(ims[0]) is np.ndarray:
#         ims = np.array(ims).astype(np.uint8)
#         if (ims.shape[-1] != 3):
#             ims = ims.transpose((0,2,3,1))
#     f = plt.figure(figsize=figsize)
#     cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
#
#     for i in range(len(ims)):
#         sp = f.add_subplot(cols, rows, i+1)
#         sp.axis('Off')
#         if titles is not None:
#             sp.set_title(titles[i], fontsize=12)
#         plt.imshow(ims[i], interpolation=None if interp else 'none')
#Check the training set (with batch of 10 as defined above
imgs, labels = next(train_generator)

#Images are shown in the output
plots(imgs, titles=labels)


augmented_images = [train_generator[0][0][0] for i in range(6)]
plotImages(imgs)

import math
STEP_SIZE_TRAIN=math.ceil(train_generator.n / train_generator.batch_size)
STEP_SIZE_VALID=math.ceil(validation_generator.n / validation_generator.batch_size)


print(train_generator.class_indices)
print(validation_generator.class_indices)


history = model.fit(
    train_generator,
    steps_per_epoch=(STEP_SIZE_TRAIN),  # 2000 images = batch_size * steps
    epochs=10,
    validation_data=validation_generator,
    validation_steps=(STEP_SIZE_VALID),  # 1000 images = batch_size * steps
    verbose=2,workers=1,use_multiprocessing=False)


labels = ['XR_ELBOW','XR_FINGER','XR_FOREARM','XR_HAND','XR_HUMERUS','XR_SHOULDER','XR_WRIST']
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
print("Confusion Matrix for valid Set:\n========================================")
Y_pred = model.predict_generator(validation_generator, validation_generator.samples // batch_size+1)
# Y_pred = model.predict(validation_generator.image_data_generator, batch_size= batch_size)
y_pred = np.argmax(Y_pred, axis=1)
cf_matrix = confusion_matrix(validation_generator.classes, y_pred)
print(cf_matrix)
import seaborn as sns
plt.figure()
sns_plot = sns.heatmap(cf_matrix, cmap='Blues')
fig = sns_plot.get_figure()
fig.savefig('/tmp/pycharm_project_27/DeepLearning/Intern/confusionMatrixForValid_new.png', dpi=400)
print('Classification Report')
print(classification_report(validation_generator.classes, y_pred, target_names=labels))
#
#
print("Confusion Matrix for Train Set:\n========================================")
Y_pred_T = model.predict_generator(train_generator, train_generator.samples // batch_size+1)
# Y_pred_T = model.predict(train_generator, batch_size= batch_size)
y_pred_T = np.argmax(Y_pred_T, axis=1)
cf_matrix_T = confusion_matrix(train_generator.classes, y_pred_T)
print(cf_matrix_T)
import seaborn as sns
plt.figure()
sns_plot_T= sns.heatmap(cf_matrix_T, annot=True,fmt='.2%', cmap='Blues')
fig_T = sns_plot_T.get_figure()
fig_T.savefig('/tmp/pycharm_project_27/DeepLearning/Intern/confusionMatrixForTrain_new.png', dpi=400)
print('Classification Report')
print(classification_report(train_generator.classes, y_pred_T, target_names=labels))

import matplotlib.image  as mpimg

plt.figure()
#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) # Get number of epochs

# ------------------------------------------------
# Plot training and validation accuracy per epoch
# ------------------------------------------------
plt.figure()
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
# plt.legend()
plt.savefig('/tmp/pycharm_project_27/DeepLearning/Intern/losses_new.png')

