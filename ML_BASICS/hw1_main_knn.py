
from load_mnist import * 
import hw1_knn  as mlBasics  
import numpy as np 
from random import sample
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

# Load data - ALL CLASSES
x, y = load_mnist('training'  )
X_test, y_test = load_mnist('testing'   )

#print(np.shape(y_train))

#stratSplit = StratifiedShuffleSplit(y, n_iter=1, test_size=0.5, random_state=42)
sp = StratifiedShuffleSplit(n_splits=1, test_size=(59/60), random_state=42)
for train_index, _ in sp.split(x, y):
    x_train, y_train = x[train_index], y[train_index]

# Reshape the image data into rows  
x_train = np.reshape(x_train, (x_train.shape[0], -1))
x = np.reshape(x, (x.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

dists =  mlBasics.compute_euclidean_distances(x_train,X_test) 

y_test_pred_three = mlBasics.predict_labels(dists, y_train, k=1) 
y_test_pred_five = mlBasics.predict_labels(dists, y_train, k=5) 

print ('{:,.2f}'.format(np.mean(y_test_pred_three==y_test)*100), "of test examples classified correctly for k=1.")
print ('{:,.2f}'.format(np.mean(y_test_pred_five==y_test)*100), "of test examples classified correctly for k=5.")

def VisualizeMnist():
    minInRowsOne = np.argmin(dists, axis=1)

    top_train_one = x_train[minInRowsOne[:10]]

    f, axarr = plt.subplots(1, 10)
    plt.title("k = 1")
    for i in range(0,10):
        first_image = top_train_one[i]
        #print(np.shape(first_image))
        first_image = np.array(first_image, dtype='uint8')
        pixels = first_image.reshape((28, 28))
        axarr[i].imshow(pixels, cmap='gray')
        axarr[i].set_axis_off()
        #axa.axis('off')
        #z += 1
        #plt.pause(.1)
    #plt.tight_layout()
    plt.show()
    plt.clf()

    plt.close(fig=1)   
    minInRowsFive = np.argpartition(dists, 5, axis=1)[:10]
    #print("lengt is ",len(minInRowsFive))

    #fOne, axarrF = plt.subplots(10, 6)
    #plt.title("k = 5")
    #print(minInRowsFive.shape())
    fOne, axarrF = plt.subplots(10, 5)
    #lengt = len(minInRowsFive) - 1
    #print("lengt ", lengt)
    for r in range(0,10):
        print("r is",r)
        top_train_five = x_train[minInRowsFive[r][:5]]

        plt.title("k = 5")

        for i in range(0,5):
            first_image = top_train_five[i]
            #print(np.shape(first_image))
            first_image = np.array(first_image, dtype='uint8')
            pixels = first_image.reshape((28, 28))
            axarrF[r][i].imshow(pixels, cmap='gray')
            axarrF[r][i].set_axis_off()
        #axa.axis('off')
        #z += 1
        #plt.pause(.1)
        #axa.axis('off')
        #z += 1
        #plt.pause(0.5)
    plt.show()   
    plt.clf()
        
    #plt.tight_layout()
    #plt.show()
    #plt.clf()

VisualizeMnist()


print(confusion_matrix(y_test, y_test_pred_three))
print(confusion_matrix(y_test, y_test_pred_five))

def FiveFoldCrossValidation(x1, y1):
    k_range = range(1,15)
    scores = []
    kf = KFold(n_splits=5)
    for index in k_range:
        for train_index, test_index in kf.split(x_train):
            X_fun_train, x_fun_test = x1[train_index], x1[test_index]
            Y_fun_train, y_fun_test = y1[train_index], y1[test_index]

            dists_fun =  mlBasics.compute_euclidean_distances(X_fun_train,x_fun_test) 

            average = np.array([])

        
            y_test_pred_fun = mlBasics.predict_labels(dists_fun, Y_fun_train, index) 
            average = np.append(average, np.mean(y_test_pred_fun==y_fun_test)*100)
        
        #print(np.mean(average))
        scores.append(average)

    plt.plot(k_range, scores)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Testing Accuracy')
    plt.show()

    print ("the best performing k is", scores.index(max(scores))+1, "with accuracy", max(scores))

    return (scores.index(max(scores))+1)


# def FiveFoldCrossValidationKnn():
#     k_range = range(1,15)
#     scores = []
#     for k in k_range:
#         print("processing")
#         knn = KNeighborsClassifier(n_neighbors = k)

#         knn.fit(x_train, y_train)
#         #predicted = knn.predict(X_test)
#         #print(knn.score(X_test, y_test))
#         #train model with cv of 5 
#         cv_scores = cross_val_score(knn, x_train, y_train, cv=5)
#         #print(accuracy_score(y_test, predicted))
#         scores.append(np.mean(cv_scores))

#     plt.plot(k_range, scores)
#     plt.xlabel('Value of K for KNN')
#     plt.ylabel('Testing Accuracy')
#     plt.show()

#     print ("the best performing k is", scores.index(max(scores))+1, "with accuracy", max(scores))


bestPerf = FiveFoldCrossValidation(x_train, y_train)

print("for all 6000 samples - processing")

dists_full =  mlBasics.compute_euclidean_distances(x,X_test) 

# y_test_pred_one = mlBasics.predict_labels(dists, y_train, k=1) 
# y_test_pred_best = mlBasics.predict_labels(dists, y_train, k=bestPerf) 

# if(np.mean(y_test_pred_one==y_test) == np.mean(y_test_pred_best==y_test)):
#   print("k=1 performs the best")

