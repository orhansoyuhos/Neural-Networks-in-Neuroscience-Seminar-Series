# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 15:06:44 2020

@author: Orhan
"""
#import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

#import the data (be aware that the first row of the data was changed by hand before for naming columns)
raw_data = pd.read_csv('EP1.txt', sep=',')

#preprocess the data in accordance with the aim: 0 vs -1 labels; take equal number of points from each
df = pd.DataFrame(raw_data, columns = ['[channel]', '[code]', '[size]' ,'[data]'])
df_random = df.loc[df['[code]'] == -1]
df_zero = df.loc[df['[code]'] == 0] #choose the number
df_final = pd.concat([df_random, df_zero])

#create the X (features) and y (label) data
X = []
y = []
ii = 0

for i in df_final.index:
    tmp_data = df_final['[data]'][i].split(",")
    if np.shape(tmp_data)[0] == 256:
        if df_final['[code]'][i] == 0 and ii <= np.shape(df_random)[0]: #choose one label
            X.append([float(i) for i in tmp_data])
            y.append(df_final['[code]'][i])
            ii += 1
        elif df_final['[code]'][i] == -1: #choose another label
            X.append([float(i) for i in tmp_data])
            y.append(df_final['[code]'][i])

            
X = np.array(X)
y = np.atleast_2d(np.array(y)).T

#split the data into the training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1, test_size=0.1)

#normalize (scale) the data
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#train with a classifier (3 layers)
mlp = MLPClassifier( hidden_layer_sizes=(256, 128, 64), solver='sgd', learning_rate='adaptive', max_iter=200, random_state=1)
mlp.fit( X_train, y_train)

#predictions and accuracy
y_pred = mlp.predict(X_test)
mlp.score(X_test, y_test)*100

#plot the change in loss per iteration
plt.plot(mlp.loss_curve_) #mlp.loss_curve_ records loss per iteration
plt.title('Loss Curve for Unbiased Data')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()