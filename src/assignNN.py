from tensorflow.keras.layers import Dense, Conv2D, Input, UpSampling2D
from tensorflow.keras.models import Model
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
import numpy as np
import os

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# set the parameters
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
retrain = False
rescale = True
# rebalance the dataset upsampling or downsampling
# 'none', 'SMOTE', 'SMOTEENN', 'SMOTETomek', 'RandomUnderSampler'
rebalance_method = 'none'
save_model = False
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# ##################################################################################
# preprocess the dataset
# ##################################################################################

# Load the dataset covtype
from sklearn.datasets import fetch_covtype
covtype = fetch_covtype()
X = covtype.data
y = covtype.target

# transform the target variable to a one-hot encoded vector
y = tf.keras.utils.to_categorical(y-1, num_classes=7)

# get names of the features
feature_names = covtype.feature_names

# avg on 20 runs without rescale (avg time = 9min on NVIDIA 3060)
# training accuracy: 0.7490
# perfomance: mavg precision = 0.724, std = 0.014
#             wavg precision = 0.744, std = 0.010
#             mavg recall = 0.525, std = 0.009 
#             wavg recall = 0.747, std = 0.011
#             mavg f1-score = 0.576, std = 0.014
#             wavg f1-score = 0.739, std = 0.010
#             accuracy = 0.747, std = 0.011

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Normalize the dataset
if rescale:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # avg on 20 runs with rescale (avg time = 9min on NVIDIA 3060)
    # training accuracy: 0.931
    # perfomance: mavg precision = 0.897, std = 0.005
    #             wavg precision = 0.931, std = 0.003
    #             mavg recall = 0.878, std = 0.003
    #             wavg recall = 0.933, std = 0.004
    #             mavg f1-score = 0.882, std = 0.005
    #             wavg f1-score = 0.934, std = 0.004
    #             accuracy = 0.933, std = 0.004

# Apply the sampling methods only on the training set to avoid data contamination
# everything is evaluated with rescale = True
if rebalance_method == 'SMOTE':
    smote = SMOTE()
    X_train, y_train = smote.fit_resample(X_train, y_train)
    # avg on 10 runs with SMOTE (avg time = 34min on NVIDIA 3060)
    # training accuracy: 0.93
    # perfomance: mavg precision = 0.822, std = 0.006
    #             wavg precision = 0.914, std = 0.004
    #             mavg recall = 0.943, std = 0.004
    #             wavg recall = 0.912, std = 0.004
    #             mavg f1-score = 0.868, std = 0.005
    #             wavg f1-score = 0.909, std = 0.003
    #             accuracy = 0.912, std = 0.004
elif rebalance_method == 'SMOTEENN':
    smote_enn = SMOTEENN()
    X_train, y_train = smote_enn.fit_resample(X_train, y_train)
    # avg on 10 runs with SMOTEENN (avg time = 31min on NVIDIA 3060)
    # training accuracy: 0.97
    # perfomance: mavg precision = 0.821, std = 0.008
    #             wavg precision = 0.908, std = 0.004
    #             mavg recall = 0.932, std = 0.004
    #             wavg recall = 0.904, std = 0.004
    #             mavg f1-score = 0.866, std = 0.007
    #             wavg f1-score = 0.901, std = 0.004
    #             accuracy = 0.904, std = 0.004
elif rebalance_method == 'SMOTETomek':
    smote_tomek = SMOTETomek()
    X_train, y_train = smote_tomek.fit_resample(X_train, y_train)
    # avg on 10 runs with SMOTETomek (avg time = 37min on NVIDIA 3060)
    # training accuracy: 0.96
    # perfomance: mavg precision = 0.825, std = 0.011
    #             wavg precision = 0.923, std = 0.003
    #             mavg recall = 0.938, std = 0.003
    #             wavg recall = 0.912, std = 0.003
    #             mavg f1-score = 0.870, std = 0.007
    #             wavg f1-score = 0.907, std = 0.003
    #             accuracy = 0.912, std = 0.003
elif rebalance_method == 'RandomUnderSampler':
    rus = RandomUnderSampler()
    X_train, y_train = rus.fit_resample(X_train, y_train)
    # avg on 20 runs with RandomUnderSampler (avg time = 4min on NVIDIA 3060)
    # training accuracy: 0.89
    # perfomance: mavg precision = 0.611, std = 0.021
    #             wavg precision = 0.780, std = 0.005
    #             mavg recall = 0.843, std = 0.005
    #             wavg recall = 0.729, std = 0.005
    #             mavg f1-score = 0.678, std = 0.009
    #             wavg f1-score = 0.742, std = 0.005
    #             accuracy = 0.729, std = 0.005

# ##################################################################################
# Create the neural network model
# ##################################################################################

if retrain == False and 'modelNN' in os.listdir():
    model = tf.keras.models.load_model('modelNN/modelNN.h5')
    print('-----------------------------------------------------------------')
    print('Model loaded, set retrain to True to retrain the model')
    print('-----------------------------------------------------------------')
else:
    # Create a neural network model
    model = tf.keras.models.Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(54, activation='relu'),
        Dense(512, activation='hard_silu'),
        # reshape the input to 3D 8x8x8
        tf.keras.layers.Reshape((8, 8, 8)),
        # upsampling layer
        UpSampling2D(size=(2, 2)),
        # convolutional layer
        Conv2D(32, kernel_size=2, activation='relu'),
        # pooling layer
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # flatten the output
        tf.keras.layers.Flatten(),
        Dense(128, activation='relu'),
        Dense(7, activation='softmax')
    ])

    # create the directory to save the model
    import os
    if not os.path.exists('modelNN'):
        os.makedirs('modelNN')

    # save the image of the model, disalbe if graphviz is not installed
    #tf.keras.utils.plot_model(model, to_file='modelNN/modelNN.png', show_shapes=True)
    model.summary()

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=15, batch_size=64)

    # save the model
    if save_model:
        model.save('modelNN/modelNN.h5')

# save the image of the model, disalbe if graphviz is not installed
#tf.keras.utils.plot_model(model, to_file='modelNN/modelNN.png', show_shapes=True)

# Evaluate the model
model.evaluate(X_test, y_test)

# confusion matrix
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 8))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, norm=matplotlib.colors.LogNorm(), cmap='Blues',annot_kws={"size": 12})
plt.xlabel('Predicted', fontsize=20)
plt.ylabel('Actual', fontsize=20)
plt.show()

# classification report
from sklearn.metrics import classification_report
# names of the classes
target_names = ['Spruce/Fir', 'Lodgepole Pine', 'Ponderosa Pine', 'Cottonwood/Willow', 'Aspen', 'Douglas-fir', 'Krummholz']
print('-----------------------------------------------------------------')
print('Classification report')
print('-----------------------------------------------------------------')
print(classification_report(y_test, y_pred, target_names=target_names, digits=5))
print('-----------------------------------------------------------------')