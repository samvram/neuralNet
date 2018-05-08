from sklearn.neighbors import KNeighborsClassifier
import pickle

with open('data.pickle','rb') as f:
    data = pickle.load(f)

Xtrain = data['Xtrain']
Ytrain = data['Ytrain']
Xtest = data['Xtest']
Ytest = data['Ytest']

def knn(k, Xtrain, Ytrain):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(Xtrain, Ytrain)
    return knn

if __name__ == '__main__':
    knn = knn(5, Xtrain, Ytrain)
    acc = knn.score(Xtest, Ytest)
    print('The accuracy from knn with n=5 is : '+str(acc*100)+' %')