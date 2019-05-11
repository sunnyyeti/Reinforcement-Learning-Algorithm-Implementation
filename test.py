from sklearn.cluster import KMeans,MiniBatchKMeans
import numpy as np
Samples = 50*(10**6)
Features = 100
X = np.empty(Samples*Features, dtype=np.float16)
for i in range(0,Samples*Features,Samples):
    X[i:i+Samples] = np.random.random(Samples)
X=X.reshape((Samples,Features))

clusters = 500000
Centroids = np.empty(clusters*Features,dtype=np.float16)
for i in range(0, clusters*Features, clusters):
    Centroids[i:i+clusters] = np.random.random(clusters)
Centroids = Centroids.reshape((clusters,Features))
import time
from tqdm import tqdm
ITER = 300
for _ in (range(ITER)):
    #start = time.time()
    new_centroids = np.zeros(Samples,dtype=np.int64)
    for i in tqdm(range(Samples)):
        cur_sample = X[i,:]
        cur_distance = cur_sample.dot(Centroids.T)
        new_centroids[i] = cur_distance.argmin()
    for c in range(clusters):
        Centroids[c,:] = X[new_centroids==c].mean(axis=0)
    #end = time.time()
    #print(end-start)