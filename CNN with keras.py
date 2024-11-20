import tensorflow as tf
from tensorflow.keras import models,layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

#upload_mnist

(x_train,y_train),(x_test,y_test) = mnist.load_data()

#pishpardazesh

#normalization
x_train = x_train / 255.0
x_test = x_test / 255.0

#reshape 
#(high,wight,channel)
x_train = x_train.reshape((x_train.shape[0],28,28,1))
x_test = x_test.reshape((x_test.shape[0],28,28,1))

#One-Hot Encoding
y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)

#generate model 

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

#compile models

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#train models

history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

#evaluate models

test_loss, test_acc = model.evaluate(x_test, y_test)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_acc}")










