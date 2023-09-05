import json
import numpy as np
import os
import shutil

with open('./train.json', 'r') as fopen:
    a = json.load(fopen)

q = []
for i in os.listdir('./train'):
    q.append(i)

map = {j: i for (i, j) in enumerate(q)}

if not os.path.exists('./train_train/'):
    os.mkdir('./train_train/')
    os.mkdir('./train_test/')

with open('./newdset.txt', 'w') as fopen:
    for i in a['images']:
        id = i['id']
        test = 1 if np.random.rand() < 0.1 else 0
        folder = i['file_name'].split('_')[0]
        filename = i['file_name']
        label = map[folder]
        fopen.write(f'{id}\t{test}\t{label}\t{folder}\t{filename}\n')
        if test:
            if not os.path.exists(f'./train_test/{folder}'):
                os.mkdir(f'./train_test/{folder}')
            shutil.copy(f'./train/{folder}/{filename}', f'./train_test/{folder}/')
        else:
            if not os.path.exists(f'./train_train/{folder}'):
                os.mkdir(f'./train_train/{folder}')
            shutil.copy(f'./train/{folder}/{filename}', f'./train_train/{folder}/')