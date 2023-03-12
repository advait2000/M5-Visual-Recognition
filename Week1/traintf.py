# Import required packages
import tensorflow as tf

tf.random.set_seed(42)
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from resnet_tf import ResNet
from model_tf import build, build_deep

train_data_dir = '/Users/advaitdixit/Documents/Masters/dataset/MIT_split/train'
test_data_dir = '/Users/advaitdixit/Documents/Masters/dataset/MIT_split/test'
# Initialize training and testing directories
# train_data_dir = 'MIT_small_train_1_comb_classes/train'
# test_data_dir = 'MIT_small_train_1_comb_classes/test'

# Initialize batch parameters
img_width = 224
img_height = 224
batch_size = 16
number_of_epoch = 100
learning_rate = 0.001
weight_decay = 0

# Perform data augmentation
datagen = ImageDataGenerator(featurewise_center=True,
                             samplewise_center=False,
                             featurewise_std_normalization=True,
                             samplewise_std_normalization=False,
                             rotation_range=15.,
                             width_shift_range=0.,
                             height_shift_range=0.,
                             shear_range=5.,
                             zoom_range=[0.5, 5],
                             channel_shift_range=0.,
                             fill_mode='nearest',
                             cval=0.,
                             horizontal_flip=True,
                             vertical_flip=False,
                             rescale=0,
                             validation_split=0.1)

# Initialize train generator
train_generator = datagen.flow_from_directory(train_data_dir,
                                              target_size=(img_width, img_height),
                                              batch_size=batch_size,
                                              class_mode='categorical',
                                              subset='training')

# Initialize test generator
test_generator = datagen.flow_from_directory(test_data_dir,
                                             target_size=(img_width, img_height),
                                             batch_size=batch_size,
                                             class_mode='categorical')

# Initialize validation generator
validation_generator = datagen.flow_from_directory(train_data_dir,
                                                   target_size=(img_width, img_height),
                                                   batch_size=batch_size,
                                                   class_mode='categorical',
                                                   subset='validation')


model = build(224, 224, 8, 'elu')
opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

# model = tf.keras.models.experimental.SharpnessAwareMinimization(model)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

print("Model parameters", model.count_params())

checkpoint = ModelCheckpoint("model1_bigdataset.hdf5", monitor="val_accuracy", save_best_only=True, verbose=1)
callbacks = [checkpoint]

# Train the model
history = model.fit(train_generator,
                    steps_per_epoch=(int(train_generator.samples // batch_size)),
                    epochs=number_of_epoch,
                    validation_data=validation_generator,
                    validation_steps=(int(validation_generator.samples // batch_size)), callbacks=callbacks)

print(history.history.keys())
print(max(history.history["val_accuracy"]))

# Summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('resnettfacc.jpg')
plt.close()

# Summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('resnettfloss.jpg')
