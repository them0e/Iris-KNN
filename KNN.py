"""

 M.Sh

    
    
"""
import pandas as pd
import numpy as np
import math
import operator

def euclidianDistance(data1, data2, length):
    distance = 0
    for x in range(length):
        distance += np.square(data1[x] - data2[x])
       
    return np.sqrt(distance)


def knn(trainingSet, testInstance, k):
 
    distances = {}
    sort = {}
    length = testInstance.shape[0]
    print(length)
    
    
    # Calculating euclidean distance between each row of training data and test data
    for x in range(len(trainingSet)):
        
       
        dist = euclidianDistance(testInstance, trainingSet[x], length)
        distances[x] = dist
       
 
    
    # Sorting them on the basis of distance
    sorted_d = sorted(distances.items(), key=operator.itemgetter(1)) #by using it we store indices also
    sorted_d1 = sorted(distances.items())
    print(sorted_d[:5])
    print(sorted_d1[:5])
   
 
    neighbors = []
    
   
    
    # Extracting top k neighbors
    for x in range(k):
        neighbors.append(sorted_d[x][0])
        counts = {"Iris-setosa":0,"Iris-versicolor":0,"Iris-virginica":0}
    
    
    # Calculating the most freq class in the neighbors
    for x in range(len(neighbors)):
        response = trainingSet[neighbors[x]][-1]
 
        if response in counts:
            counts[response] += 1
        else:
            counts[response] = 1
  
    print(counts)
    sortedVotes = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)
    print(sortedVotes)
    return(sortedVotes[0][0], neighbors)
    
    

# Preprocessing Input data
data = pd.read_csv('iris.csv')
X = data.iloc[:, 1:-1].values
Y = data.iloc[:, 5].values

trainperc=0.2

# Split and encode labels
def split(dataset,trainprec):
        
    dataset = dataset.sample(frac=1) #shuffling data
    X = dataset.iloc[:, 1:-1].values #features
    y = dataset.iloc[:, 5].values #labels
    #trainprec = float(trainprec)
    #trainperc = testperc - 1
    trainlength = len(X) * trainperc
    #testlength = len(X) * testperc
    
    from sklearn.preprocessing import LabelEncoder 
    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y)
    
    #TRAIN
    X_train = X[int(trainlength):,:]
    y_train = y[int(trainlength):]
    #TEST
    X_test = X[:int(trainlength),:]
    y_test = y[:int(trainlength)]    
    
    return X_train, y_train, X_test, y_test


X_train, y_train, X_test, y_test = split(data,trainperc)

#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


def compute_confusion_matrix(true, pred):
  '''Computes a confusion matrix using numpy for two np.arrays
  true and pred.

  Results are identical (and similar in computation time) to: 
    "from sklearn.metrics import confusion_matrix"

  However, this function avoids the dependency on sklearn.'''

  K = len(np.unique(true)) # Number of classes 
  result = np.zeros((K, K))

  for i in range(len(true)):
    result[true[i]][pred[i]] += 1

  return result

# To round up the predicted values to 3 unique values or 0,1 and 2 
def r_pred(result):
    for i in range(len(result)):
        if result[i] >= 1.5:
               result[i] = 2 
        elif result[i] < 1.5 and result[i] >= 1: 
               result[i]=1
        elif result[i] < 1: 
               result[i]=0
    return result

# To transform the predicted values from 0,1 and 2 to Iris-setosa, Iris-versicolor and Iris-virginica
def r_predvals(result):
    testRes = {}
    for i in range(len(result)):
        if result[i] == 2:
               testRes[i] = 'Iris-virginica'
        elif result[i] < 1.5 and result[i] >= 1: 
               testRes[i]= 'Iris-versicolor'
        elif result[i] < 1: 
               testRes[i]= 'Iris-setosa' 
    return testRes


confm = compute_confusion_matrix(y_test, testpred)


result = {}
neigh = {}
for i in range(len(X_test)):
    result[i],neigh[i] = knn(X_train, X_test[i], 3) # here we gave k=3
    
result1 = {}
neigh1 = {}
for i in range(len(X_test)):
    result1[i],neigh1[i] = knn(X_train, X_test[i], 1) # here we gave k=1

result6 = {}
neigh6 = {}
for i in range(len(X_test)):
    result6[i],neigh6[i] = knn(X_train, X_test[i], 6) # here we gave k=6


result = r_pred(result)           
result1 = r_pred(result1)
result6 = r_pred(result6)




def to1d(result):
    #lists
    temp = []
    dictList = []
            
    for key, value in result.items():
        temp = [value]
        dictList.append(temp)
    
    dictList = np.asarray(dictList)
    dictList = np.squeeze(dictList)
    return dictList



setosa = X[:30,:]
versicolor = X[50:80,:]
virginica = X[110:140,:]

# Import Gaussian Naive Bayes model to compare it with our KNN Algorithm
from sklearn.naive_bayes import GaussianNB

# Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets
model.fit(X_train,y_train)


# Predict output
predicted1 = model.predict(setosa)
predicted2 = model.predict(versicolor) 
predicted3 = model.predict(virginica) 
predicted4 = model.predict(X_test) 

confusion_matrixB1 = pd.crosstab(y_test, predicted1, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrixB1)
confusion_matrixB2 = pd.crosstab(y_test, predicted2, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrixB2)
confusion_matrixB3 = pd.crosstab(y_test, predicted3, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrixB3)
confusion_matrixB4 = pd.crosstab(y_test, predicted4, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrixB4)




result = to1d(result)
result1 = to1d(result1)
result6 = to1d(result6)



confusion_matrix = pd.crosstab(y_test, result, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)
confusion_matrix1 = pd.crosstab(y_test, result1, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix1)
confusion_matrix6 = pd.crosstab(y_test, result6, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix6)



#from sklearn.neighbors import KNeighborsClassifier
#knn1 = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
#knn1.fit(X_train, y_train)
#
#y_pred = knn1.predict(X_test)


FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)  
print ('FP: ' ,FP)
FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
print ('FN : ', FN)
TP = np.diag(confusion_matrix)
print ('TP: ',TP)
TN = confusion_matrix.values.sum() - (FP + FN + TP)
print ('TN: ',TN)

# Sensitivity, hit rate, recall, or true positive rate
recall = TP/(TP+FN)
print('Recall: ',recall)
prec = TP/(TP+FP)
print('Precision: ', prec)
acc = (TP+TN)/(TP+FP+FN+TN)
print('Accuracy: ',acc)

