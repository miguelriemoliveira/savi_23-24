#!/usr/bin/env python3


import glob
import json
from sklearn.model_selection import train_test_split


def main():

    # -----------------------------------------------------------------
    # Prepare Datasets
    # -----------------------------------------------------------------
    data_path = '/home/mike/savi_datasets/dogs-vs-cats/train/'
    image_filenames = glob.glob(data_path + '*.jpg')
    # To test the script in good time, select only 1000 of the 25000 images

    # Use a rule of 70% train, 20% validation, 10% test

    train_filenames, remaining_filenames = train_test_split(image_filenames, test_size=0.3)
    validation_filenames, test_filenames = train_test_split(remaining_filenames, test_size=0.33)

    print('We have a total of ' + str(len(image_filenames)) + ' images.')
    print('Used ' + str(len(train_filenames)) + ' train images')
    print('Used ' + str(len(validation_filenames)) + ' validation images')
    print('Used ' + str(len(test_filenames)) + ' test images')

    d = {'train_filenames': train_filenames,
         'validation_filenames': validation_filenames,
         'test_filenames': test_filenames}

    json_object = json.dumps(d, indent=2)

    # Writing to sample.json
    with open("dataset_filenames.json", "w") as outfile:
        outfile.write(json_object)


if __name__ == "__main__":
    main()
