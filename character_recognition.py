import os
import gc
import cv2
import caer
import canaro
import numpy as np
from tensorflow.keras.utils import to_categorical


def main():
    IMG_SIZE = (80, 80)
    channels = 1
    char_path = r'../input/the-simpsons-characters-dataset/simpsons_dataset'

    char_dict = {}
    for char in os.listdir(char_path):
        char_dict[char] = len(os.listdir(os.path.join(char_path, char)))
        
    # Sort in descending order
    char_dict = caer.sort_dict(char_dict, descending=True)

    # Get top 10 characters 
    characters = []
    count = 0
    for i in char_dict:
        characters.append(i[0])
        count += 1
        if count >= 10:
            break

    # Create the training data
    train = caer.preprocess_from_dir(
        char_path,
        characters,
        channels=channels,
        IMG_SIZE=IMG_SIZE,
        isShuffle=True
    )

    # Separate feature set and labels
    featureSet, labels = caer.sep_train(train, IMG_SIZE=IMG_SIZE)

    # Normalize the featureSet in (0, 1)
    featureSet = caer.normalize(featureSet)
    labels = to_categorical(labels, len(characters))

    x_train, x_test, y_train, y_test = caer.train_val_split(
        featureSet,
        labels,
        val_ratio=0.2
    )

    del train
    del featureSet
    del labels
    gc.collect()

    BATCH_SIZE = 32
    EPOCHS = 10

    # Create new data generator
    data_gen = canaro.generators.imageDataGenerator()
    train_gen = data_gen.flow(
        np.array(x_train),
        np.array(y_train),
        batch_size=BATCH_SIZE
    )

    # Create a model
    model = canaro.models.createSimpsonsModel(IMG_SIZE=IMG_SIZE,
                                            channels=channels,
                                            output_dim=len(characters),
                                            loss='binary_crossentropy',
                                            decay=1e-6,
                                            learning_rate=0.001,
                                            momentum=0.9,
                                            nesterov=True
                                            )


if __name__ == "__main__":
    main()