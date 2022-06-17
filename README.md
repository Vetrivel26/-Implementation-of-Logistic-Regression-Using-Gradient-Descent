# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1.Read the given dataset.
2.Fitting the dataset into the training set and test set.
3.Applying the feature scaling method.
4.Fitting the logistic regression into the training set.
5.Prediction of the test and result
6.Making the confusion matrix
7.Visualizing the training set results.

## Program:
```
/*
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by : VETRIVEL S
RegisterNumber :  212221240060
*/

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

datasets = pd.read_csv("/content/Social_Network_Ads (1).csv")
X = datasets.iloc[:,[2,3]].values
Y = datasets.iloc[:,[4]].values

from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_X

X_Train = sc_X.fit_transform(X_Train)
X_Test = sc_X.fit_transform(X_Test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_Train, Y_Train)

Y_Pred = classifier.predict(X_Test)
Y_Pred

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_Test, Y_Pred)
cm

from sklearn import metrics
accuracy = metrics.accuracy_score(Y_Test, Y_Pred)
accuracy

recall_sensitivity = metrics.recall_score(Y_Test, Y_Pred, pos_label = 1)
recall_specificity = metrics.recall_score(Y_Test, Y_Pred, pos_label = 0)
recall_sensitivity, recall_specificity

from matplotlib.colors import ListedColormap
X_Set, Y_Set = X_Train, Y_Train
X1,X2 = np.meshgrid(np.arange(start = X_Set[:,0].min()-1, stop = X_Set[:,0].max()+1, step = 0.01), 
                    np.arange(start = X_Set[:,1].min()-1, stop = X_Set[:,1].max()+1, step = 0.01))

plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),
X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red','green')))

plt.xlim(X1.min(), X2.max())
plt.ylim(X1.min(), X2.max())
for i,j in enumerate(np.unique(Y_Set)):
    plt.scatter(X_Set[Y_Set == j,0],X_Set[Y_Set==j,1],
                c=ListedColormap(('red','green'))(i),label=j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.label('Estimated Salary')
plt.legend()
plt.show()
 
*/
```

## Output:
![logistic regression using gradient descent](sam.png)
![173594360-52e1fb7b-bcda-47ce-9816-e73a95e9a6b8](https://user-images.githubusercontent.com/95363138/174242893-e885d304-d2f7-4cf7-a37f-aab73be2f06f.jpg)
![173594449-b36c4c0a-efec-4d8f-b7a6-ad302ca15cd5](https://user-images.githubusercontent.com/95363138/174242918-dd9dc5ff-7ec9-4791-9403-2a4a3b53c9b6.jpg)
### Accuracy
![3](https://user-images.githubusercontent.com/95363138/174242931-21be1d61-ce37-4343-a66f-7b68e819f1bc.jpg)
### Recalling Sensitivity and Specificity:
![4](https://user-images.githubusercontent.com/95363138/174242959-2eed736b-cc04-4100-81cd-25bef166e0da.jpg)


### Visulaizing Training set Result:
![output](https://user-images.githubusercontent.com/95363138/174242820-a10bb5d3-592f-4cc3-8584-766cb3bd680a.jpeg)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

