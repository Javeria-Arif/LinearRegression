import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


'Opening the data file'
data = pd.read_csv('dataset.csv')
print(data.shape)       #returns rows and columns
#data.head(6)          Shows the data table upto nth values


'Choosing the desired columns'
X = data.iloc[:, 2:3].values
Y = data.iloc[:, 3:4].values
#print(X)
#print (Y)

#plt.scatter(X,Y, color='red')


'Splitting my data into training and test samples'
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.2,random_state = 10)
#print(len(X_test))

'Applying Linear Regression'
from sklearn.linear_model import LinearRegression
reg = LinearRegression()

reg.fit(X_train,Y_train)        #Training data

y_pred = reg.predict(X_test)    #Predicting results


'Visualizing trained data'
plt.scatter(X_train, Y_train, color = 'green')
plt.plot(X_train, reg.predict(X_train), color = 'red')
plt.xlabel('Head size(in cms)')
plt.ylabel('Brain weight(in grams)')
plt.title('Head size VS Brain weight (Training set)')
plt.show()

'Visualizing tested data'
plt.scatter(X_test, Y_test, color = 'blue')
plt.plot(X_train, reg.predict(X_train), color = 'red')
plt.xlabel('Head size(in cms)')
plt.ylabel('Brain weight(in grams)')
plt.title('Head size VS Brain weight (test set)')
plt.show()

print(reg.score(X_test,Y_test))               #Gives the accuracy of prediction