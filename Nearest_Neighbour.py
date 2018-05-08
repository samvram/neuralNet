import pickle
import numpy as np

with open('data.pickle','rb') as f:
    data = pickle.load(f)

Xtrain = data['Xtrain']
Ytrain = data['Ytrain']
Xtest = data['Xtest']
Ytest = data['Ytest']

Ypredict = []
Ypredict_2 = []
acc = 0
acc_2 = 0
for j in range(0,len(Xtest)):
    test_case = Xtest[j]
    distance = np.zeros((len(Xtrain),1))
    distance_2 = np.zeros((len(Xtrain),1))
    for i in range(0,len(Xtrain)):
        distance[i] = np.sum(np.abs(np.array(Xtrain[i])-np.array(test_case)))
        distance_2[i] = np.sqrt(np.sum(np.square(np.array(Xtrain[i])-np.array(test_case))))
        # print('In train case : %d; in test case : %d; distance : %d, true matches : %d' % (i, j, distance[i], acc))
    min_index = np.argmin(distance)
    min_index_2 = np.argmin(distance_2)
    Ypredict.append(Ytrain[min_index])
    Ypredict_2.append(Ytrain[min_index_2])
    print("%d predicted value : %s, true value : %s"%(j, Ytrain[min_index],Ytest[j]))
    print("%d predicted value : %s, true value : %s"%(j, Ytrain[min_index_2],Ytest[j]))
    print()
    if Ytest[j]==Ytrain[min_index]:
        print('True Match')
        acc += 1
    if Ytest[j]==Ytrain[min_index_2]:
        print('True Match')
        acc_2 += 1



accuracy = acc/len(Xtest)*100.0
accuracy_2 = acc_2/len(Xtest)*100.0
print('The accuracy of L1 norm is '+str(accuracy)+' %')
print('The accuracy of L2 norm is '+str(accuracy_2)+' %')
