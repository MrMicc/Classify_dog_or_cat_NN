from utils import datasets, plt_images, model
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True' #this is necessary since the OpenMP bring a issue to run multiple process
dts = datasets.DogsAndCats(batch_size=100, img_shape=150)
epoch = 10

total_train = len(os.listdir(dts.train_dir['dogs'])) + len(os.listdir(dts.train_dir['cats']))

total_valid = len(os.listdir(dts.valid_dir['dogs'])) + len(os.listdir(dts.valid_dir['cats']))
print('Total of Images in the dataset: \n',
      'Train dog image: {}\n'.format(len(os.listdir(dts.train_dir['dogs']))),
      'Train cat image {}\n'.format(len(os.listdir(dts.train_dir['cats']))),
      'Validation dog images: {}\n'.format(len(os.listdir(dts.valid_dir['dogs']))),
      'validation cat images: {}\n'.format(len(os.listdir(dts.valid_dir['cats'])))
      )


nn = model.NN()

#print(nn.summary())

#if you want to see examples of images, use the function below
#plt_images.show_samples(dts.valid_data)



history = nn.train(data_train=dts.train_data, epoch=epoch, data_validation=dts.valid_data, total_train=total_train,
         total_valid=total_valid, batch_size=dts.batch_size)


print(history.history)

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt_images.show_history(accuracy=accuracy, val_accuracy=val_accuracy, loss=loss, val_loss=val_loss, epoch_range=range(epoch))


