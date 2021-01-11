import os
import gc
import cv2
import caer
import canaro
import numpy as np


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


if __name__ == "__main__":
    main()