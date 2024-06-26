import numpy as np
from tensorflow.keras.layers import Dense, Conv2D, Input
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.models import Model
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import StandardScaler
import logging
import os
logging.getLogger('tensorflow').setLevel(logging.DEBUG)

##################################################################################
# make the smallest neural network possible to get weighted accuracy of 90% or more
##################################################################################

# !!!! set retrain to True to retrain the model !!!!
retrain = False
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Load the dataset covtype
covtype = fetch_covtype()
X = covtype.data
y = covtype.target

# transform the target variable to a one-hot encoded vector
y = tf.keras.utils.to_categorical(y-1, num_classes=7)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Normalize the dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

if retrain == False and 'modelNNSmall' in os.listdir():
    model = tf.keras.models.load_model('modelNNSmall/NNnonQuantized.h5')
    print('-----------------------------------------------------------------')
    print('Model loaded, set retrain to True to retrain the model')
    print('-----------------------------------------------------------------')
else:
    # Create a neural network model with mixed inputs
    # first branch of the model branch x
    inputAll = Input(shape=(54,), name='inputAll')
    x = Dense(45, activation='hard_silu')(inputAll)
    x = Dense(125, activation='relu')(x)
    # reshape the input to 3D 5x5x5
    x = tf.keras.layers.Reshape((5, 5, 5))(x)
    x = UpSampling2D(size=(2, 2))(x)
    # convolutional layer
    x = Conv2D(10, kernel_size=2, activation='relu6')(x)
    # pooling layer
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    # flatten the output
    x = tf.keras.layers.Flatten()(x)
    x = Model(inputs=inputAll, outputs=x)

    # input again the first 14 values of the features and concatenate with the output of the flatten layer
    # second branch of the model branch y
    inputf14 = Input(shape=(14,), name='inputf14')
    y = Model(inputs=inputf14, outputs=inputf14)

    # combine the two branches of the model branch z
    combined = tf.keras.layers.concatenate([x.output, y.output])
    z = Dense(12, activation='relu')(combined)
    z = Dense(7, activation='softmax')(z)
    model = Model(inputs=[x.input, y.input], outputs=z)

    # Compile the model
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit([X_train, X_train[:, 0:14]], y_train, epochs=20, batch_size=64)

    # save the model to disk
    model.save("modelNNSmall/NNnonQuantized.h5")

    # quantize the model
    model.export("test", "tf_saved_model")
    # convert the model to tflite
    converter = tf.lite.TFLiteConverter.from_saved_model("test")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
    tflite_model = converter.convert()
    with open("modelNNSmall/NN.tflite", "wb") as f:
        # write the model to disk
        f.write(tflite_model)

##################################################################################
# Evaluate the models 
##################################################################################

#!!! non-quantized model !!!
model.summary()
model.evaluate([X_test, X_test[:, 0:14]], y_test)

# image of the model, disable this line if you don't have graphviz installed
#tf.keras.utils.plot_model(model, to_file='modelNNSmall/model.png', show_shapes=True)

# classification report
from sklearn.metrics import classification_report
# names of the classes
target_names = ['Spruce/Fir', 'Lodgepole Pine', 'Ponderosa Pine', 'Cottonwood/Willow', 'Aspen', 'Douglas-fir', 'Krummholz']
y_pred = model.predict([X_test, X_test[:, 0:14]])
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)
print('-----------------------------------------------------------------')
print('Classification report for non-quantized model')
print('-----------------------------------------------------------------')
print(classification_report(y_test, y_pred, target_names=target_names, digits=5))
print('-----------------------------------------------------------------')

# confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import confusion_matrix
plt.figure(figsize=(12, 8))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, norm=matplotlib.colors.LogNorm(), cmap='Blues',annot_kws={"size": 12})
plt.xlabel('Predicted', fontsize=20)
plt.ylabel('Actual', fontsize=20)
plt.show()


##################################################################################

#!!! quantized model !!!
interpreter = tf.lite.Interpreter(model_path="modelNNSmall/NN.tflite")

# allocate memory for the input and output tensors
interpreter.allocate_tensors()

# get the input and output tensors
input_details = interpreter.get_input_details()

# get the output tensor
output_details = interpreter.get_output_details()

# convert the data to the right format float32
input_shape1 = input_details[0]['shape']
input_shape2 = input_details[1]['shape']
input_data1 = np.array(X_test, dtype=np.float32)
input_data2 = np.array(X_test[:, 0:14], dtype=np.float32)

# make predictions
for i in range(0, len(X_test)):
    interpreter.set_tensor(input_details[0]['index'], input_data1[i:i+1])
    interpreter.set_tensor(input_details[1]['index'], input_data2[i:i+1])
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    y_pred[i] = np.argmax(output_data)

# classification report
print('-----------------------------------------------------------------')
print('Classification report for quantized model')
print('-----------------------------------------------------------------')
print(classification_report(y_test, y_pred, target_names=target_names, digits=5))
print('-----------------------------------------------------------------')