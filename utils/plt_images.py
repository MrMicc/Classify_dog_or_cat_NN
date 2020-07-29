import matplotlib.pyplot as plt
import os


def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()


def show_samples(data_set, number_samples=5):
    imgs, _ = next(data_set)[:number_samples]
    plotImages(imgs)


def show_history(accuracy, val_accuracy, loss, val_loss, epoch_range):
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    show(range=epoch_range, value=accuracy, value_2=val_accuracy, labels=['Training Accuracy', 'Validation Accuracy'],
         legend_position='upper right', title='Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    show(range=epoch_range, value=loss, value_2=val_loss, labels=['Traning Loss', 'Validation Loss'],
         legend_position='upper right', title='Training and Validation Loss')
    plt.savefig('./foo.png')
    plt.show()


def show(range, value, labels, value_2, legend_position='', title=''):
    plt.plot(range, value, label=labels[0])
    plt.plot(range, value_2, label=labels[1])
    plt.legend(loc=legend_position)
    plt.title(title)
