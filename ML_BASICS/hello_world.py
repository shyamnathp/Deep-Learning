import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy import stats
import matplotlib.pyplot as plt

X = np.array([[5, 1, 0, 2], [10, 2, 8, 7]])
labels = np.array([0,0,1,0])
euclidean_distances(X, X)

minInRows = np.argpartition(X, 1, axis=1)

#print(len(minInRows))
for r in range(len(minInRows)):
    print(labels[minInRows[r][:3]])
for r in range(len(minInRows)):
    m = stats.mode(labels[minInRows[r][:3]])
    print(m[0])

#dim=np.shape(X)
#print(dim[1])
pred_labels = np.array([])
#pred_labels = np.append(pred_labels, 2)

#for i in minInRows:
for r in range(len(minInRows)):
 m = stats.mode(labels[minInRows[r][:3]])
 pred_labels = np.append(pred_labels, m[0])   

print(pred_labels)
#listOfCordinates = list(result[1])
#for cord in minInRows:
#    print(cord)
%matplotlib inline
x = np.arange(0, 5, 0.1)
y = np.sin(x)
plt.plot(x, y)
#plt.show()