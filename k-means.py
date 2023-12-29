
"""
Created on Thu Dec 14 19:59:53 2023

@author: GulerKandeger
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

dataset = pd.read_csv('iris_dataset.csv')

x = dataset.iloc[:,[0,1,2,3]].values

#en doğru cluster sayısını bulmak için wcss değerlerine bakılır
wcss=[]
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++',random_state=123)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)  

plt.figure(figsize=(8, 4))    
plt.plot(range(1,11), wcss)    


kmeans = KMeans(n_clusters= 3 , init='k-means++')
y_pred=kmeans.fit_predict(x)


plt.figure(figsize=(8, 4))
plt.title('K-Means')
plt.scatter(x[y_pred==0,0],x[y_pred==0,1],s=100,c='red',label='Iris-Setosa')
plt.scatter(x[y_pred==1,0],x[y_pred==1,1],s=100,c='blue',label='Iris-Versicolor')
plt.scatter(x[y_pred==2,0],x[y_pred==2,1],s=100,c='green',label='Iris-Virginica')





