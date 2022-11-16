"""
Author: Robert van der Klis

What does this module do

Usage: python3 ...
"""


# Import statements
import os

# Function definitions
def main():
    ids = []
    with open ('/home/robert/projects/part_detection/happyWhale/train.csv', 'r') as fopen:
        for line in fopen:
            img, id = line.split(',')
            if id.startswith('w_'):
                ids.append(id)

    with open('train.csv', 'w') as fopen:
        idx = 0
        fopen.write('Image,Id\n')
        for file in os.listdir('/home/robert/projects/part_detection/occlusion/train'):
            if not ',' in file:
                fopen.write(f'{file},{ids[idx]}\n')
                idx += 1



if __name__ == "__main__":
    main()
