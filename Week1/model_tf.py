from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization


def build(width, height, classes, activation):
    # initialize the input shape to be "channels last" and the
    # channels dimension itself
    inputShape = (width, height, 3)
    input_tensor = keras.Input(shape=inputShape)
    list_model = []
    for i in range(1):
        x = Conv2D(32, (3, 3), input_shape=inputShape)(input_tensor)
        x = BatchNormalization(axis=-1)(x)
        x = Activation(activation)(x)
        x0 = x
        x = layers.DepthwiseConv2D(kernel_size=3, padding="same")(x)
        x = Activation(activation)(x)
        x = layers.Add()([x, x0])  # Residual.
        x = layers.Conv2D(30, kernel_size=1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(32, (3, 3))(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation(activation)(x)
        x0 = x
        x = layers.DepthwiseConv2D(kernel_size=3, padding="same")(x)
        x = Activation(activation)(x)
        x = layers.Add()([x, x0])  # Residual.
        x = layers.Conv2D(30, kernel_size=1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(64, (3, 3))(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation(activation)(x)
        x0 = x
        x = layers.DepthwiseConv2D(kernel_size=3, padding="same")(x)
        x = Activation(activation)(x)
        x = layers.Add()([x, x0])  # Residual.
        x = layers.Conv2D(30, kernel_size=1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        list_model.append(x)

    x = layers.Add()(list_model)
    x = layers.GlobalAvgPool2D()(x)
    x = Flatten()(x)  # this converts our 3D feature maps to 1D feature vectors
    x = Dense(256, activation=activation)(x)
    x = Dropout(0.1)(x)
    x = Dense(classes, activation='softmax')(x)

    model = Model(input_tensor, x)
    return model


def build_deep(width, height, classes, activation):
    # initialize the input shape to be "channels last" and the
    # channels dimension itself
    inputShape = (width, height, 3)
    input_tensor = keras.Input(shape=inputShape)
    list_model = []
    for i in range(1):
        x = Conv2D(32, (3, 3), input_shape=inputShape)(input_tensor)
        x = BatchNormalization(axis=-1)(x)
        x = Activation(activation)(x)
        x0 = x
        x = layers.DepthwiseConv2D(kernel_size=3, padding="same")(x)
        x = Activation(activation)(x)
        x = layers.Add()([x, x0])  # Residual.
        x = layers.Conv2D(64, kernel_size=1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(32, (3, 3))(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation(activation)(x)
        x0 = x
        x = layers.DepthwiseConv2D(kernel_size=3, padding="same")(x)
        x = Activation(activation)(x)
        x = layers.Add()([x, x0])  # Residual.
        x = layers.Conv2D(64, kernel_size=1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(64, (3, 3))(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation(activation)(x)
        x0 = x
        x = layers.DepthwiseConv2D(kernel_size=3, padding="same")(x)
        x = Activation(activation)(x)
        x = layers.Add()([x, x0])  # Residual.
        x = layers.Conv2D(128, kernel_size=1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(128, (3, 3))(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation(activation)(x)
        x0 = x
        x = layers.DepthwiseConv2D(kernel_size=3, padding="same")(x)
        x = Activation(activation)(x)
        x = layers.Add()([x, x0])  # Residual.
        x = layers.Conv2D(256, kernel_size=1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        list_model.append(x)

    x = layers.Add()(list_model)
    x = layers.GlobalAvgPool2D()(x)
    x = Flatten()(x)  # this converts our 3D feature maps to 1D feature vectors
    x = Dense(256, activation=activation)(x)
    x = Dropout(0.1)(x)
    x = Dense(classes, activation='softmax')(x)

    model = Model(input_tensor, x)
    return model


model = build_deep(224, 224, 3, activation="elu")
print(model.summary())
