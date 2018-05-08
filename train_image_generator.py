import pickle
import numpy as np
from PIL import Image

file_list = ['cifar/data_batch_1','cifar/data_batch_2','cifar/data_batch_3','cifar/data_batch_4','cifar/data_batch_5','cifar/test_batch']


with open('cifar/batches.meta','rb') as f:
    label_dict = pickle.load(f,encoding='bytes')

test_labels = [x.decode('utf-8') for x in label_dict[b'label_names']]

data = []
for file in file_list:
    with open(file,'rb') as f:
        data.append(pickle.load(f, encoding='bytes'))

count = dict()
for label in test_labels:
    count[label] = 0

Xtrain = []
Ytrain = []

for dat in data:
    for i in range(0,dat[b'data'].shape[0]):
        Xtrain.append(dat[b'data'][i,:])
        Ytrain.append(test_labels[dat[b'labels'][i]])
        count[test_labels[dat[b'labels'][i]]] += 1

Xtest = Xtrain[50000:]
Ytest = Ytrain[50000:]

Xtrain = Xtrain[:50000]
Ytrain = Ytrain[:50000]

print('The shape of each unit in Xtrain is '+str(Xtrain[0].shape))
print('The size of Xtrain is '+str(len(Xtrain)))
print('The size of Xtest is '+str(len(Xtest)))
print('The first value of Ytrain is '+Ytrain[0])

# Portion for extracting as image

rgb = np.zeros((32,32,3), dtype='uint8')
for i in range(0,len(Xtrain)):
    train_image = Xtrain[i]
    rgb = np.reshape(train_image, (32,32,3), 'F')
    rgb[: ,: ,0] = np.rot90(rgb[:, : ,0])
    rgb[: ,: ,1] = np.rot90(rgb[:, : ,1])
    rgb[: ,: ,2] = np.rot90(rgb[:, : ,2])
    rgb[:, :, 0] = np.rot90(rgb[:, :, 0])
    rgb[:, :, 1] = np.rot90(rgb[:, :, 1])
    rgb[:, :, 2] = np.rot90(rgb[:, :, 2])
    rgb[:, :, 0] = np.rot90(rgb[:, :, 0])
    rgb[:, :, 1] = np.rot90(rgb[:, :, 1])
    rgb[:, :, 2] = np.rot90(rgb[:, :, 2])
    image = Image.fromarray(rgb, 'RGB')
    image.save('train\Train_image_'+str(i)+'_'+Ytrain[i]+'.png')



