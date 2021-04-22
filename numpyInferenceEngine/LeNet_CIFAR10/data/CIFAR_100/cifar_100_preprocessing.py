'''
Download and preprocess CIFAR-100 dataset.
Please first download the CIFAR-100 dataset from "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz".
Then, decompress the dataset.
Finally, run this python script.

By Boyuan Feng
'''

import numpy as np
import pickle

def unpickle(file):
  fo = open(file, 'rb')
  dict = pickle.load(fo, encoding = 'bytes')
  fo.close()
  if b'data' in dict:
    dict[b'data'] = dict[b'data'].reshape((-1, 3, 32, 32)) #.swapaxes(1, 3).swapaxes(1, 2)
  return dict

def load_data_one(f):
  batch = unpickle(f)
  data = batch[b'data']
  labels = batch[b'fine_labels']
  return data, labels

def load_data(files, data_dir, label_count):
  data, labels = load_data_one(data_dir + '/' + files[0])
  for f in files[1:]:
    data_n, labels_n = load_data_one(data_dir + '/' + f)
    data = np.append(data, data_n, axis=0)
    labels = np.append(labels, labels_n, axis=0)
  return data, labels

train_files = ['train']
test_files = ['test']

data_dir = 'cifar-100-python'

'''
batch = unpickle('data_batch_1')
batch.keys()
dict_keys([b'batch_label', b'labels', b'data', b'filenames'])
'''

train_X, train_Y = load_data(train_files, data_dir, 100)
test_X, test_Y = load_data(test_files, data_dir, 100)

with open('train_X.npy', 'wb') as f:
    np.save(f, train_X)
with open('train_Y.npy', 'wb') as f:
    np.save(f, train_Y)
    

with open('test_X.npy', 'wb') as f:
    np.save(f, test_X)
with open('test_Y.npy', 'wb') as f:
    np.save(f, test_Y)

