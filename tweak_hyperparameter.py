import pickle
from k_nn import knn
from matplotlib import pyplot as plt
import numpy as np

with open('data.pickle','rb') as f:
    data = pickle.load(f)

Xtrain = data['Xtrain']
Ytrain = data['Ytrain']
Xtest = data['Xtest']
Ytest = data['Ytest']

accuracy = []
for k in range(5,50):
    knn = knn(k, Xtrain[:40000], Ytrain[:40000])
    accuracy.append(knn.score(Xtrain[40000:],Ytrain[40000:]))

plt.plot(range(5,50),accuracy)

print('They maximum accuracy is for k = %d'%(5+np.argmax(accuracy)))

