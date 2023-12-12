#!/usr/bin/env python3


import glob
from sklearn.model_selection import train_test_split
from dataset import Dataset
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

from model import Model
from trainer import Trainer


def main():

    # -----------------------------------------------------------------
    # Hyperparameters initialization
    # -----------------------------------------------------------------
    batch_size = 100
    learning_rate = 0.001
    num_epochs = 10

    # -----------------------------------------------------------------
    # Create model
    # -----------------------------------------------------------------
    model = Model()

    # -----------------------------------------------------------------
    # Prepare Datasets
    # -----------------------------------------------------------------
    data_path = '/home/mike/savi_datasets/dogs-vs-cats/train/'
    image_filenames = glob.glob(data_path + '*.jpg')
    # To test the script in good time, select only 1000 of the 25000 images
    image_filenames = image_filenames[0:5000]

    train_filenames, validation_filenames = train_test_split(image_filenames, test_size=0.2)

    print('We have a total of ' + str(len(image_filenames)) + ' images. Used '
          + str(len(train_filenames)) + ' for training and ' + str(len(validation_filenames)) +
          ' for validation.')

    train_dataset = Dataset(train_filenames)
    validation_dataset = Dataset(validation_filenames)

    # Try the train_dataset
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True)

    # Just for testing the train_loader
    tensor_to_pil_image = transforms.ToPILImage()

#     for batch_idx, (inputs, labels) in enumerate(train_loader):
#         print('batch_idx = ' + str(batch_idx))
#         print('inputs shape = ' + str(inputs.shape))
#
#         model.forward(inputs)
#
#         image_tensor_0 = inputs[0, :, :, :]
#         print(image_tensor_0.shape)
#
#         image_pil_0 = tensor_to_pil_image(image_tensor_0)
#         print(type(image_pil_0))
#
#         fig = plt.figure()
#         plt.imshow(image_pil_0)
#         plt.show()
#
#         exit(0)

    # -----------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------
    trainer = Trainer(model=model,
                      train_loader=train_loader,
                      validation_loader=validation_loader,
                      learning_rate=learning_rate,
                      num_epochs=num_epochs)
    trainer.train()

    plt.show()


if __name__ == "__main__":
    main()
