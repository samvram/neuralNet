import pickle
import numpy as np
from PIL import Image

file_list_train = ['cifar/data_batch_1','cifar/data_batch_2','cifar/data_batch_3','cifar/data_batch_4','cifar/data_batch_5']
file_list_test = ['cifar/test_batch']

with open('cifar/batches.meta','rb') as f:
    label_dict = pickle.load(f,encoding='bytes')
    
label_list = label_dict[b'label_names']
for x in label_list:
    x = x.decode('utf-8')
    
train_list = []
for file in file_list_train:
    with open(file,'rb') as f:
        train_list.append(pickle.load(f,encoding='bytes'))
        
    
test_list = []
for file in file_list_test:
    with open(file,'rb') as f:
        test_list.append(pickle.load(f,encoding='bytes'))
        
test_case = test_list[0]
file_names = test_case[b'filenames']
list_of_test_images = []

for i in range(0,len(test_case[b'data'])):
    x = dict()
    x['file_name'] = (file_names[i]).decode('utf-8')
#    print(x['file_name'])
    x['data'] = test_case[b'data'][i]
    x['label'] = label_list[int(test_case[b'labels'][i])]
    list_of_test_images.append(x)

# For seeing a single image
for i in range(0, len(list_of_test_images)):
    imag = list_of_test_images[i]
    image_array = imag['data']
    image_array = np.reshape(image_array, (32,32,3), 'F')
    image_array[:,:,0] = np.rot90(image_array[:,:,0])
    image_array[:,:,1] = np.rot90(image_array[:,:,1])
    image_array[:,:,2] = np.rot90(image_array[:,:,2])
    image_array[:,:,0] = np.rot90(image_array[:,:,0])
    image_array[:,:,1] = np.rot90(image_array[:,:,1])
    image_array[:,:,2] = np.rot90(image_array[:,:,2])
    image_array[:,:,0] = np.rot90(image_array[:,:,0])
    image_array[:,:,1] = np.rot90(image_array[:,:,1])
    image_array[:,:,2] = np.rot90(image_array[:,:,2])
    img = Image.fromarray(image_array, 'RGB')
    file_name = 'test\\' + imag['label'].decode('utf-8') + '_' + imag['file_name']
    print(file_name)
    img.save(file_name)
#
