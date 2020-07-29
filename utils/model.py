import tensorflow as tf
import numpy as np

class NN():

    def __init__(self):
        self.model = tf.keras.models.Sequential([
            #1fst block of convolutional
            tf.keras.layers.Conv2D(32, (3,3), activation='relu',input_shape=(150,150,3)),
            tf.keras.layers.MaxPooling2D(2,2),

            #2sec block of convolutional
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),

            #2sec block of convolutional
            tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),

            #2sec block of convolutional
            tf.keras.layers.Conv2D(128,(3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),

            #flatten the results from convolution
            tf.keras.layers.Flatten(),
            #feeding the model with 512 neurons on Dense layer
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(2)
        ])

        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    def summary(self):
        return self.model.summary()


    def train(self, data_train, data_validation, total_train, total_valid, batch_size, epoch:int = 100):
        return self.model.fit_generator(data_train,
                                        steps_per_epoch=int(np.ceil(total_train / float(batch_size))),
                                        epochs=epoch,
                                        validation_data=data_validation,
                                        validation_steps=int(np.ceil(total_valid/ float(batch_size)))
                                        )