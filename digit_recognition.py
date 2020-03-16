# Classification template

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')


from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
dataset = onehotencoder.fit_transform(dataset).toarray()

X = dataset[:, 10:]
y = dataset[:, :10]
x_test = pd.read_csv('test.csv')

# Fitting classifier to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X, y)

# Predicting the Test set results
y_pred = classifier.predict(x_test)

#DECODING
result = []

for i in range(28000):
    for j in range(10):
        if y_pred[i, j] == 1:
            result.append(j)
            break
        
#Test cases
dataset3 = pd.read_csv('sample_submission.csv')
   
#saving the submission file  
for i in range(28000):
    dataset3.iloc[i, 1] = result[i]       
            
dataset3.to_csv('result.csv')
            
            