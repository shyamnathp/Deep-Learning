from load_mnist import * 
import numpy as np 
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

# Load data - ALL CLASSES
X_train, y_train = load_mnist('training'  )
X_test, y_test = load_mnist('testing'   )

# Reshape the image data into rows  
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

model = Sequential()

model.add(Dense(units=32, activation='sigmoid', input_shape=(784,)))
model.add(Dense(units=10,activation='softmax'))
#model.summary()

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=128, epochs=5, verbose=False, validation_split=.1)
loss, accuracy  = model.evaluate(X_test, y_test, verbose=False)

print(f'Classification accuracy: {accuracy:.3}')

plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('iterations/epochs')
plt.show()


