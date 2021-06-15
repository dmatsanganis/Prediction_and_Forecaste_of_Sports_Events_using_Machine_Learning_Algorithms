import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.regularizers import l2
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Function, which creates the neurons.
def init(x_train,y_train,x_test,y_test):

    # Create neuron model via keras library.
    model = keras.Sequential()
    model.add(layers.Dense(4*28, input_dim=28, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4*28,activation='relu'))           
    model.add(layers.Dense(2*28, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))

    # kernel_regularizer=l2(0.0001),
    # activity_regularizer=l2(0.01)

    # Tranforms the variables to an acceptable format.
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")

    # Compile and fit the keras model.
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics= ['accuracy'])
    history = model.fit(x_train, y_train, batch_size=512, epochs=1000, verbose=1, validation_data = (x_test, y_test))

    # Put all the model's predictions into a table.
    predictions = model.predict(x_test).round()

    # Initialize the rights and wrong - predicted 
    # matches'counter.
    rights_counter = 0
    wrongs_counter = 0

    # A For-Loop, which gets the right 
    # and the wrong - predicted matches.
    for i,j in enumerate(predictions):

        # If the match is predicted correctly.
        if((y_test[i] == j).all()):
            rights_counter+=1

        # Else If the match is predicted incorrectly.
        else:
            wrongs_counter+=1

    # Making Multi-Layer Neural Networw Plots.
    # Summarize history for Accuracy.
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Multi-Layer Neural Network Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train Accuracy', 'Test Accuracy'], loc='upper left')
    plt.show()

    # Summarize history for loss.
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Multi-Layer Neural Network Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train Accuracy', 'Test Accuracy'], loc='upper right')
    plt.show()

    # Console's Message, regarding the Wrong - Predicted Matches appears.
    print('Wrong - Predicted Matches: '+ str(wrongs_counter))
    # Console's Message, regarding the Right - Predicted Matches appears.
    print('Rights - Predicted Matches: '+ str(rights_counter))
    # Console's Message, regarding the Predicted Matches' Accuracy stat appears.
    print('Predicted Matches Accuracy: ' + str(rights_counter/(wrongs_counter + rights_counter)))