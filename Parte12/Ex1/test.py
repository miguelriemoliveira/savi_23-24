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

import torch.nn.functional as F


def main():

    # -----------------------------------------------------------------
    # Hyperparameters initialization
    # -----------------------------------------------------------------
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

    test_filenames = dataset_filenames['test_filenames']
    test_filenames = test_filenames[0:100]

    print('Used ' + str(len(test_filenames)) + ' for testing ')

    test_dataset = Dataset(test_filenames)

    batch_size = len(test_filenames)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    # Just for testing the train_loader
    tensor_to_pil_image = transforms.ToPILImage()

    # -----------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Load the trained model
    checkpoint = torch.load('models/checkpoint.pkl')
    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)
    model.eval()  # we are in testing mode
    batch_losses = []
    for batch_idx, (inputs, labels_gt) in enumerate(test_loader):

        # move tensors to device
        inputs = inputs.to(device)
        labels_gt = labels_gt.to(device)

        # Get predicted labels
        labels_predicted = model.forward(inputs)

    # Transform predicted labels into probabilities
    predicted_probabilities = F.softmax(labels_predicted, dim=1).tolist()
    # print(predicted_probabilities)

    probabilities_dog = [x[0] for x in predicted_probabilities]

    # print(probabilities_dog)

    # Make a decision using the largest probability
    predicted_is_dog = [x > 0.5 for x in probabilities_dog]
    print('predicted_is_dog=' + str(predicted_is_dog))

    labels_gt_np = labels_gt.cpu().detach().numpy()
    ground_truth_is_dog = [x == 0 for x in labels_gt_np]
    print('ground_truth_is_dog=' + str(ground_truth_is_dog))

    # labels_predicted_np = labels_predicted.cpu().detach().numpy()
    # print('labels_gt_np = ' + str(labels_gt_np))
    # print('labels_predicted_np = ' + str(labels_predicted_np))

    # Count FP, FN, TP, and TN
    TP, FP, TN, FN = 0, 0, 0, 0
    for gt, pred in zip(ground_truth_is_dog, predicted_is_dog):

        if gt == 1 and pred == 1:  # True positive
            TP += 1
        elif gt == 0 and pred == 1:  # False positive
            FP += 1
        elif gt == 1 and pred == 0:  # False negative
            FN += 1
        elif gt == 0 and pred == 0:  # True negative
            TN += 1

    print('TP = ' + str(TP))
    print('TN = ' + str(TN))
    print('FP = ' + str(FP))
    print('FN = ' + str(FN))

    # Compute precision and recall
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision*recall)/(precision+recall)

    print('Precision = ' + str(precision))
    print('Recall = ' + str(recall))
    print('F1 score = ' + str(f1_score))

    # Show image
    # inputs = inputs.cpu().detach()
    # print(inputs)

    fig = plt.figure()
    idx_image = 0
    for row in range(4):
        for col in range(4):
            image_tensor = inputs[idx_image, :, :, :]
            image_pil = tensor_to_pil_image(image_tensor)
            print('ground_truth is dog = ' + str(ground_truth_is_dog[idx_image]))
            print('predicted is dog = ' + str(predicted_is_dog[idx_image]))

            ax = fig.add_subplot(4, 4, idx_image+1)
            plt.imshow(image_pil)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])

            text = 'GT '
            if ground_truth_is_dog[idx_image]:
                text += 'is dog'
            else:
                text += 'is not dog'

            text += '\nPred '
            if predicted_is_dog[idx_image]:
                text += 'is dog'
            else:
                text += 'is not dog'

            if ground_truth_is_dog[idx_image] == predicted_is_dog[idx_image]:
                color = 'green'
            else:
                color = 'red'

            ax.set_xlabel(text, color=color)

            idx_image += 1

    plt.show()


    # plt.show()
if __name__ == "__main__":
    main()
