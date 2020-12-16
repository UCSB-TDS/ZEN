import os
import imageio
import cv2
import numpy as np


train_path = '/home/boyuan/ZKP/TinyQuantCNN/numpyInferenceEngine/LeNet_ORL/orl_face_dataset/data/'
train_dirs = os.listdir(train_path)


train_X = np.zeros((360, 1, 56, 46))
train_Y = np.zeros((360, ))
test_X = np.zeros((40, 1, 56, 46))
test_Y = np.zeros((40, ))


idx = 0
train_idx = 0
test_idx = 0
for dir in train_dirs:
    path = train_path + dir
    files = os.listdir(path)
    for file in files:
        file_path = path + '/' + file
        img = imageio.imread(file_path)
        img = cv2.resize(img, dsize=(56, 46), interpolation=cv2.INTER_CUBIC)
        img = img.reshape((1, 56,46))
        label = int(dir[1:])
        if idx % 10 == 0:
            test_Y[test_idx] = label-1
            test_X[test_idx] = img
            test_idx += 1
        else:
            train_Y[train_idx] = label-1
            train_X[train_idx] = img
            train_idx += 1
        idx += 1

print("train_X.shape: ", train_X.shape, ", train_Y.shape: ", train_Y.shape)
print("test_X.shape: ", test_X.shape, ", test_Y.shape: ", test_Y.shape)



with open('train_X.npy', 'wb') as f:
    np.save(f, train_X)
with open('train_Y.npy', 'wb') as f:
    np.save(f, train_Y)
    

with open('test_X.npy', 'wb') as f:
    np.save(f, test_X)
with open('test_Y.npy', 'wb') as f:
    np.save(f, test_Y)




