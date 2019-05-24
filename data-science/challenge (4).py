# -*- coding: utf-8 -*-
"""
Created on Tue May 21 09:27:14 2019

@author: alyssa rose
"""

# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


# reading in a unit to gain an understanding
# of what the data looks like
data = pd.read_csv('unit0012_rms.csv', index_col = "timestamp")
alarm = pd.read_csv('unit0012_alarms.csv', header = None, names = ["timestamp", "warning"], 
                    index_col = "timestamp")

# using the describe function to understand the data; if there is a significant
# drop or increase in values, it might be the failure signal 
desc = data.describe()
"""
Objective 1: Smoothing the data

Despite the signal processing libraries available, given that the data is fairly
constant (as shown in the small difference between 25% and 75% quartile), it doesn't
make sense to try to fit a higher order curve as it will result in data loss or overfitting.

To handle the issue, the data will be smoothed via a moving avg filter.
This will take a subset of the data along the curves, calculate the avg and update
the given point (the mid of the min and max of the subset) such that the point
is updated symmetrically.

limitations: ************
"""

# filtering the "nonsense" data within reasonable bounds
# still allows for negative values which may indicate failure
lower = desc.iloc[4] - (desc.iloc[4] + 0.1*desc.iloc[5])
upper = desc.iloc[6] + desc.iloc[5]
for j in range(len(data)):
        m = data.iloc[j] > upper 
        h = data.iloc[j] < lower
        if max(m):
            data.iloc[j] = desc.iloc[4]
        elif max(h):
            data.iloc[j] = desc.iloc[6]
            
# handling standard noise with a rolling mean
for i in range(5):
    data.iloc[:, i] = data.iloc[:, i].rolling(window=3).mean()
    

data.iloc[78000:79176].plot(ylim=(-25, 1400))


"""
There is heavy correlation between the two temperature, and motor_voltage/ current.
RPM and motor_current seem to be somewhat correlated, and
RPM appears to have no correlation with the temperatures

Despite correlation, nothing can be said of their dependence
"""
# noticeable trend, tend to increase together
y = data.loc[:, "rpm"]
a = data.loc[:, "motor_current"]
plt.scatter(a ,y)

# noticeable trend, tend to increase together
y = data.loc[:, "rpm"]
a = data.loc[:, "motor_voltage"]
plt.scatter(a ,y)

# very correlated, increase together
y = data.loc[:, "motor_temp"]
a = data.loc[:, "inlet_temp"]
plt.scatter(a ,y)

# extremely correlated
y = data.loc[:, "motor_voltage"]
a = data.loc[:, "motor_current"]
plt.scatter(a ,y)

# somewhat correlated
y = data.loc[:, "motor_voltage"]
a = data.loc[:, "inlet_temp"]
plt.scatter(a ,y)

# slightly more correlated
y = data.loc[:, "motor_voltage"]
a = data.loc[:, "motor_temp"]
plt.scatter(a ,y)

# seemingly no correlation
y = data.loc[:, "rpm"]
a = data.loc[:, "inlet_temp"]
plt.scatter(a ,y)

# seemingly no correlation
y = data.loc[:, "rpm"]
a = data.loc[:, "motor_temp"]
plt.scatter(a ,y)

"""
The warning system: roughly 2 hours behind when a dip in RPM occurs

Failure signal: RPM becomes negative or close to 0 the day before or
day of unit failure; general trend is that the motor current
and motor voltage values increase over time
"""



"""
Objective 2: Predict the 30 models that will fail

This is a standard Classification problem, either the unit fails
or it does not.
"""

"""
Unsupervised Learning: K-means clustering
Advantage: unsupervised
Disadvantage: the training data is one cluster (all units that fail)
while the test data is comprised of two clusters, those that fail and those that
don't, so the model is not sure what second cluster data looks like before the 
test set, and mis-labels the training data
"""
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

from sklearn.cluster import KMeans
clus = KMeans(n_clusters = 2, init = "k-means++")

new_data = []
k = 0
# reading in the data
# Replace with directory to training set
for file in os.listdir(r"C:\Users\alyss\OneDrive\Documents\College\Challenge\train"):
    if file.endswith("_rms.csv"):
        data = pd.read_csv(file)
        desc = data.describe()
        
        lower = desc.iloc[4, 0] - (desc.iloc[4, 0] + 0.1*desc.iloc[5, 0])
        upper = desc.iloc[6, 0] + desc.iloc[5, 0]
        for j in range(len(data)):
            m = data.iloc[j, 1] > upper 
            h = data.iloc[j, 1] < lower
            if m:
                data.iloc[j, 1] = desc.iloc[4, 1]
            elif h:
                data.iloc[j, 1] = desc.iloc[6, 1]
        length = len(data)
        data = data.iloc[(length-450):length, 1]
        new_data.append(data)
new_data_fit = sc.fit_transform(new_data)
clus.fit(new_data_fit)

df = pd.DataFrame()
pred = []

# Replace with directory to test set
for file in os.listdir(r"C:\Users\alyss\OneDrive\Documents\College\Challenge\test"):
    if file.endswith("_rms.csv"):
        test = pd.read_csv(file)
        desc = test.describe()
        
        lower = desc.iloc[4, 0] - (desc.iloc[4, 0] + 0.1*desc.iloc[5, 0])
        upper = desc.iloc[6, 0] + desc.iloc[5, 0]
        for j in range(len(data)):
            m = test.iloc[j, 1] > upper 
            h = test.iloc[j, 1] < lower
            if m:
                test.iloc[j, 1] = desc.iloc[4, 1]
            elif h:
                test.iloc[j, 1] = desc.iloc[6, 1]
        length = len(test)
        test = test.iloc[(length-450):length, 1]
        pred.append(test)
prediction = clus.predict(pred) 


"""
Classifying with ANN

Advantages: supervised learning (tells the model before hand that all the training
data is of one type)

Disadvantages: usuallly reserved for higher dimensional data, might be prone to 
overfitting
"""
y_train = pd.DataFrame(index = range(20), columns = range(1))
for x in range(20):
    y_train.iloc[x] = 1
    
"""
commented out lines were tests done to see if the ANN could predict accurately on
a small portion of the training data itself; which it was able to classify correctly.
This was done as a "wellness" check of the model to see if it was performing as 
expected (as there is not a test file to compare against for correctness)
"""
# y_train = y_train.iloc[0:16,:]
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
new_data = np.array(new_data)

# new_test = new_data[16:20,:]
# new_data = new_data[0:16,:]

new_data = sc.fit_transform(new_data)

# X_test = sc.transform(new_test)

X_test = sc.transform(pred)

import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 213, init = 'uniform', activation = 'relu', input_dim = 450))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 213, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(new_data, y_train, batch_size = 5, nb_epoch = 500)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

"""
The chosen method is ANN; it predicts that units 31, 38, 40 and 44 

Problems with my method: Unsure on how to make the ANN treat all 450 samples of 
the RPM of the training data as not seperate attributes.

Issues with K-Means clustering: the training data was only of the "failed" cluster,
which means that the model will be highly innaccurate

Other Problems: 
    -determination of the "failure signal" introduced bias into the 
    system of the way that the data was handled
    
    -the training data was smaller than the test data (large issue)
    
    -it wasn't possible (for me) to encode the alarm system using OneHotEncoder
    as the time stamps weren't 1:1 matches with the rms file. The timestamp
    wasn't also easy to work with as it had more than the standard millisecond
    specification, which is a problem I didn't know how to handle (and thus 
    couldn't convert into a workable format) with Python's datetime functions
    
    -the warning system didn't seem to shed much light as it wasn't consistent.
    i.e. unit 13 had no alarms even though it failed, but it did seem to mainly go
    off within a 2 hour period after a drop in RPM (which is why my answer is
    formatted in the way that it is)
    
Challenges: 
    I have not encountered data of this form (i.e. the standard 1:1 input
    to output as the data was more structured like the entire rms file for each unit was
    the input and the knowledge that it failed or not (0 or 1) was the output). This
    was the greatest challenged faced
"""    