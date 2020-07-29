import tensorflow as tf
import os


class DogsAndCats():
    def __init__(self, batch_size: int = 100, img_shape: int = 150):
        # loading dogs and cats from google examples
        self.batch_size = batch_size
        self.img_shape = img_shape

        self.zip_dir = DogsAndCats.load('https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip')

        self.train_dir, self.valid_dir = DogsAndCats.load_dirs(self.zip_dir, 'cats_and_dogs_filtered')



        self.train_data = DogsAndCats.data_prepartion(train_data=True, batch_size=self.batch_size,
                                                                       img_shape=self.img_shape, directory=self.train_dir['base'])

        self.valid_data = DogsAndCats.data_prepartion(train_data=False, batch_size=self.batch_size,
                                                      img_shape=self.img_shape, directory=self.valid_dir['base'])



    @staticmethod
    def data_prepartion(train_data: bool = True, batch_size: int = 100, img_shape: int = 150, directory: str = ''):
        image_generation = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

        #the class_mode is hardcoded because is only possible to have to class -> Dog or Cat in this project
        return image_generation.flow_from_directory(batch_size=batch_size, directory=directory, shuffle=train_data,
                                                    target_size=(img_shape,img_shape), class_mode='binary')

    @staticmethod
    def load_dirs(zip_dir, folder):
        dir = os.path.join(os.path.dirname(zip_dir), folder)
        train_dir = os.path.join(dir, 'train')
        valid_dir = os.path.join(dir, 'validation')

        train_dir= {'base': train_dir,
                 'cats': os.path.join(train_dir, 'cats'),
                 'dogs': os.path.join(train_dir, 'dogs')}

        valid_dir= {'base': valid_dir,
                 'cats': os.path.join(valid_dir, 'cats'),
                 'dogs': os.path.join(valid_dir, 'dogs')}

        return train_dir, valid_dir


    @staticmethod
    def load(url: str):
        # the datasets will be save in a home project directory as datasets folder
        return tf.keras.utils.get_file(fname='cats_and_dogs_filterted.zip', origin=url, extract=True, cache_dir='./')
