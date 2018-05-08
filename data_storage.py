import pickle

file_list = ['cifar/data_batch_1','cifar/data_batch_2','cifar/data_batch_3','cifar/data_batch_4','cifar/data_batch_5','cifar/test_batch']


with open('cifar/batches.meta','rb') as f:
    label_dict = pickle.load(f,encoding='bytes')

test_labels = [x.decode('utf-8') for x in label_dict[b'label_names']]

data = []
for file in file_list:
    with open(file,'rb') as f:
        data.append(pickle.load(f, encoding='bytes'))

Xtrain = []
Ytrain = []

for dat in data:
    for i in range(0,dat[b'data'].shape[0]):
        Xtrain.append(dat[b'data'][i,:])
        Ytrain.append(test_labels[dat[b'labels'][i]])

Xtest = Xtrain[50000:]
Ytest = Ytrain[50000:]

Xtrain = Xtrain[:50000]
Ytrain = Ytrain[:50000]

writable = dict()
writable['Xtrain'] = Xtrain
writable['Xtest'] = Xtest
writable['Ytrain'] = Ytrain
writable['Ytest'] = Ytest

with open('data.pickle','wb') as f:
    pickle.dump(writable, f)