import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf 

#print(tf.__version__)

mnist = tf.keras.datasets.mnist  #28x28 images of hand-written digits 0-9

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#plt.imshow(x_train[1], cmap = plt.cm.binary) # just to see how images look like
#plt.show()

x_train = tf.keras.utils.normalize(x_train, axis=1)  # to scale it between 0 - 1 (normalizing)
x_test = tf.keras.utils.normalize(x_test, axis=1)   # to scale it between 0 - 1 (normalizing)
#print(x_train[0])

model = tf.keras.models.Sequential()  # starting model
model.add(tf.keras.layers.Flatten())  # input layer

model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # 1) dense layer
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # 2) dense layer

model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))  # output layer

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics = ['accuracy'])  # most complex part of neural model

# now we can start training
model.fit(x_train, y_train, epochs=2)

# calculate model validation and model accuracy
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss)
print(val_acc)

#model.save('epic_num_reader.model') # saving the model
#new_model = tf.keras.models.load_model('epic_num_reader.model')  # calling the saved model

prediction = model.predict(x_test)  # prediction
print(np.argmax(prediction[0]))




