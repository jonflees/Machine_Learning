# https://app.dominodatalab.com/u/LeJit/Clustering/view/Clustering.ipynb

from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
data = pd.read_csv("Wholesale.csv")
data.drop(["Channel", "Region"], axis = 1, inplace = True)

data = data[["Grocery", "Milk"]]
data = data.as_matrix().astype("float32", copy = False)

stscaler = StandardScaler().fit(data)
data = stscaler.transform(data)

#Plotting
"""
plt.scatter(data[:,0], data[:,1])
plt.xlabel("Groceries")
plt.ylabel("Milk")
plt.title("Wholesale Data - Groceries and Milk")
plt.savefig("results/wholesale.png", format = "PNG")
"""
dbsc = DBSCAN(eps = .5, min_samples = 15).fit(data)

labels = dbsc.labels_
core_samples = np.zeros_like(labels, dtype = bool)
core_samples[dbsc.core_sample_indices_] = True

#Extra
unique_labels = np.unique(labels)
colors = plt.cm.Spectral(np.linspace(0,1, len(unique_labels)))
for (label, color) in zip(unique_labels, colors):
    class_member_mask = (labels == label)
    xy = data[class_member_mask & core_samples]
    plt.plot(xy[:,0],xy[:,1], 'o', markerfacecolor = color, markersize = 10)
    
    xy2 = data[class_member_mask & ~core_samples]
    plt.plot(xy2[:,0],xy2[:,1], 'o', markerfacecolor = color, markersize = 5)
plt.title("DBSCAN on Wholsesale data")
plt.xlabel("Grocery (scaled)")
plt.ylabel("Milk (scaled)")
plt.savefig("results/dbscan_wholesale.png", format = "PNG")

#K Means
kmeans = KMeans(n_clusters = 2).fit(data)
labels_kmeans = kmeans.labels_
centroids = kmeans.cluster_centers_

plt.scatter(data[:,0], data[:,1], c = labels_kmeans)
plt.scatter(centroids[:,0], centroids[:,1], c = ["gold","blue"], s = 60 )

############################
from sklearn.datasets import make_moons
moons_X, moon_y = make_moons(n_samples = 2000)

def add_noise(X,y, noise_level = 0.01):
    #Number of points to be noisy
    amt_noise = int(noise_level*len(y))
    #Pick amt_noise points at random
    idx = np.random.choice(len(X), size = amt_noise)
    #Add random noise to these selected points
    noise = np.random.random((amt_noise, 2) ) -0.5
    X[idx,:] += noise
    return X

moon_noise_X = add_noise(moons_X, moon_y)
plt.scatter(moon_noise_X[:,0], moon_noise_X[:,1], c = moon_y)




############################
def makeCraters(inner_rad = 4, outer_rad = 4.5, donut_len = 2, inner_pts = 1000, outer_pts = 500):
    #Make the inner core
    radius_core = inner_rad*np.random.random(inner_pts)
    direction_core = 2*np.pi*np.random.random(size = inner_pts)
    #Simulate inner core points
    core_x = radius_core*np.cos(direction_core)
    core_y = radius_core*np.sin(direction_core)
    crater_core = zip(core_x, core_y)
    #Make the outer ring
    radius_ring = outer_rad + donut_len*np.random.random(outer_pts)
    direction_ring = 2*np.pi*np.random.random(size = outer_pts)
    #Simulate ring points
    ring_x = radius_ring*np.cos(direction_ring)
    ring_y = radius_ring*np.sin(direction_ring)
    crater_ring = zip(ring_x, ring_y)
    
    return np.array(crater_core), np.array(crater_ring)

crater_core, crater_ring = makeCraters(inner_pts = 2000, outer_pts = 1000)

plt.scatter(crater_core[:,0], crater_core[:,1], c = "red")
plt.scatter(crater_ring[:,0], crater_ring[:,1], c = "blue")
plt.rc('font', family='sans-serif') 
plt.title("Crater Dataset")

def knn_estimate(data, k, point):
    n,d = data.shape
    #Reshape the datapoint, so the cdist function will work
    point = point.reshape((1,2))
    #Find the distance to the kth nearest data point
    knn = sorted(reduce(lambda x,y: x+y,cdist(data, point).tolist()))[k+1]
    #Compute the density estimate using the mathematical formula
    estimate = float(k)/(n*np.power(knn, d)*np.pi)
    return estimate

full_crater = np.vstack((crater_core, crater_ring))
knn_density = np.array([knn_estimate(full_crater, 500, point) for point in full_crater])

plt.scatter(full_crater[:,0], full_crater[:,1], s = knn_density*2000, c = knn_density*1000)
plt.title("KNN Density Estimate on Crater Dataset")



########################