
from load_mnist import * 
import keras
import numpy as np 
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras import optimizers
#%matplotlib inline

# Load data - ALL CLASSES
X_train, y_train = load_mnist('training'  )
X_test, y_test = load_mnist('testing'   )

# Reshape the image data into rows  
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

#normalizing to [0,1]
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print(X_train.shape)
#converting to binary class matrix - one coded vectors 
#since we are dealing with categorical data
y_train = keras.utils.to_categorical(y_train, num_classes=10)   #10 is num_classes
y_test = keras.utils.to_categorical(y_test, num_classes=10)

#model.summary()

def full_operation(n, regular, hidden, inputSize, Xtrain, Xtest, ytrain, ytest, output):

    model = Sequential()
    model.add(Dense(units=hidden, activation='relu', input_shape=(inputSize,), kernel_regularizer=regular, bias_regularizer=regular))
    model.add(Dense(units=hidden,activation='relu', kernel_regularizer=regular, bias_regularizer=regular))
    model.add(Dense(units=output,activation='softmax'))
    print(model.summary)
    optimize =  optimizers.SGD(lr=0.01, nesterov=n)
    model.compile(optimizer= optimize, loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(Xtrain, ytrain, batch_size=128, epochs=25, verbose=False, validation_split=.1)
    loss, accuracy  = model.evaluate(Xtest, ytest, verbose=False)

    print(f'Classification accuracy: {accuracy:.3}')
    return history

def OperateAndPlot(hidden, inputSize, Xtrain, Xtest, ytrain, ytest, output):
    historyAll = []
    historyAll.append(full_operation(False, keras.regularizers.l1(0), hidden, inputSize, Xtrain, Xtest, ytrain, ytest, output))
    historyAll.append(full_operation(True, keras.regularizers.l1(0), hidden, inputSize, Xtrain, Xtest, ytrain, ytest, output))
    historyAll.append(full_operation(True, keras.regularizers.l1(0.001), hidden, inputSize, Xtrain, Xtest, ytrain, ytest, output))
    historyAll.append(full_operation(True, keras.regularizers.l2(0.01), hidden, inputSize, Xtrain, Xtest, ytrain, ytest, output))
    # plt.plot(history.history['loss'])
    # plt.title('model loss')

    plt.plot(historyAll[0].history['loss'], 'r', label= "mlp")
    plt.plot(historyAll[1].history['loss'], 'b', label= "mlp-nesterov")
    plt.plot(historyAll[2].history['loss'], 'g', label= "mlp -l1-nesterov")
    plt.plot(historyAll[3].history['loss'], 'c', label= "mlp-l2-nesterov")
    plt.legend(loc='best')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('iterations/epochs')
    plt.show()

OperateAndPlot(512, 784, X_train, X_test, y_train, y_test, 10)

# load data
data = np.load('ORL_faces.npz')
trainX = data['trainX']
trainY = data['trainY']
testX = data['testX']
testY = data['testY']

# Reshape the image data into rows  
trainX = np.reshape(trainX, (trainX.shape[0], -1))
testX = np.reshape(testX, (testX.shape[0], -1))

#normalizing to [0,1]
trainX = trainX.astype('float32')
testX = testX.astype('float32')
trainX /= 255
testX /= 255

trainY = keras.utils.to_categorical(trainY, num_classes=20)   #10 is num_classes
testY = keras.utils.to_categorical(testY, num_classes=20)

OperateAndPlot(512, 10304, trainX, testX, trainY, testY, 20)