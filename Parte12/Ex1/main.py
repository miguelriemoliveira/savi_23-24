#!/usr/bin/env python3


import glob
import json
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
    num_epochs = 50

    # -----------------------------------------------------------------
    # Create model
    # -----------------------------------------------------------------
    model = Model()

    # -----------------------------------------------------------------
    # Prepare Datasets
    # -----------------------------------------------------------------

    with open('../Ex0_split_dataset/dataset_filenames.json', 'r') as f:
        # Reading from json file
        dataset_filenames = json.load(f)

    train_filenames = dataset_filenames['train_filenames']
    validation_filenames = dataset_filenames['validation_filenames']

    # train_filenames = train_filenames[0:1000]
    # validation_filenames = validation_filenames[0:200]

    print('Used ' + str(len(train_filenames)) + ' for training and ' + str(len(validation_filenames)) +
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
                      num_epochs=num_epochs,
                      model_path='models/checkpoint.pkl',
                      load_model=True)
    trainer.train()

    plt.show()


if __name__ == "__main__":
    main()
