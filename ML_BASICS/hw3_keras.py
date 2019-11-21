from load_mnist import * 
import keras
import numpy as np 
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras import optimizers

# Load data - ALL CLASSES
X_train, y_train = load_mnist('training'  )
X_test, y_test = load_mnist('testing'   )

# Reshape the image data into rows  
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

print(X_train.shape)
#converting to binary class matrix - one coded vectors 
#since we are dealing with categorical data
y_train = keras.utils.to_categorical(y_train, num_classes=10)   #10 is num_classes
y_test = keras.utils.to_categorical(y_test, num_classes=10)

#model.summary()

def full_operation(n, regular = keras.regularizers.l1_l2(l1 = 0.01, l2=0.01)):

    model = Sequential()
    model.add(Dense(units=512, activation='relu', input_shape=(784,), kernel_regularizer=regular, bias_regularizer=regular))
    model.add(Dense(units=512,activation='relu',  kernel_regularizer=regular, bias_regularizer=regular))
    model.add(Dense(units=10,activation='softmax'))
    optimize =  optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=n)
    model.compile(optimizer= optimize, loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, batch_size=128, epochs=25, verbose=False, validation_split=.1)
    loss, accuracy  = model.evaluate(X_test, y_test, verbose=False)

    print(f'Classification accuracy: {accuracy:.3}')
    return history

historyAll = []
historyAll.append(full_operation(False))
historyAll.append(full_operation(True))
historyAll.append(full_operation(True, keras.regularizers.l1(0.01)))
historyAll.append(full_operation(True, keras.regularizers.l2(0.01)))
# plt.plot(history.history['loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('iterations/epochs')
# plt.show()


