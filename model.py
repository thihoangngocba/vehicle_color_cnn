import tensorflow as tf
import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

class VCR_model():
  def __init__(self, input_shape, num_classes, size_se):
    super(LPCModel, self).__init__()

    self.input_shape = input_shape
    self.num_classes = num_classes
    
    #Input block
    input = keras.Input(shape=self.input_shape)

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Set aside residual
    previous_block_activation = x  

    for size in [32, 64, 128]:
    # for size in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(previous_block_activation)
        # Add back residual
        x = layers.add([x, residual])  
        # Set aside next residual
        previous_block_activation = x  

    x = layers.SeparableConv2D(size_se, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)

    output = layers.Dense(units, activation=activation)(x)

    self.model =  keras.Model(input, output)


  def load_model(self, model_path):
    try:
        self.model.load_weights(model_path)
        return True
    except:
        return False


  def imageProcessing(self, img, input_size):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = tf.cast(img, tf.float32)
    img /= 255.0
    img = tf.image.resize(img, size=input_size)
    return img

  def train_model(self, path_save, ds_train, ds_val, epochs, batch_size, lr=0.001, verbose=1):
    self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=["accuracy"])

    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=path_save,
                                                      verbose=verbose,
                                                      monitor="val_accuracy",
                                                      save_best_only=True)
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=3, min_lr=0.0000001)

    callbacks = [tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss'),
                 tf.keras.callbacks.TensorBoard(log_dir="logs"),
                 checkpointer,
                 reduce_lr]

    results = self.model.fit(ds_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks, validation_data=ds_val)
    print("Finished training")
    return results


  def model_summary(self):
    print(self.model.summary())


  def save(self, path):
    self.model.save(path)
