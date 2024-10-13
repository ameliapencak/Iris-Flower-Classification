# DataFlair Iris Flower Classification
# Import Packages
import sqlite3

#import charts

import pandas as pd
iris = pd.read_csv("data/Iris.csv",sep=',')


#split the data into training and test sets
x = iris.drop("Species", axis=1)
y = iris["Species"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)









from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)




