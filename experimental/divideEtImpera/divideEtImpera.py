from sklearn.ensemble import BaggingClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Conv2D, Dropout
from tensorflow.keras.layers import UpSampling2D
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np

class divideEtImpera:
    def __init__(self, X_train, y_train):
        # Initialize any necessary variables or attributes here
        self.X_train = X_train
        self.y_train = y_train
        
        # neural network model
        self.modelNN = Sequential([
            Input(shape=(X_train.shape[1],)),
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

        # neural network model to split the data
        self.modelNNSplitter = Sequential([
            Input(shape=(X_train.shape[1],)),
            Dense(128, activation='hard_silu'),
            Dense(64, activation='sigmoid'),
            Dense(2, activation='softmax')
        ])

        # BaggingClassifier model
        self.modelML = BaggingClassifier(n_estimators=20, random_state=42)

    
    def fit(self, iterations=10, leaky=0.1, leakyTolerance=0.1):
        # fitting logic
        # X: input features
        # y: input labels
        # 1. (only first loop) split data into 2 parts (1 to neural network, 2 to BaggingClassifier)
        # 2. (by-pass first loop) train a very simple MLM on to choose wich part of the data 
        #    to send to the neural network and wich to the BaggingClassifier
        # 3. train the neural network with the first part
        # 4. train the BaggingClassifier with the second part
        # 5. check wich part of the data is right and send the wrong part to the other model

        #####################################################################################################
        # 1. (only first loop) split data into 2 parts (1 to neural network, 2 to BaggingClassifier)
        #####################################################################################################
        X_trainNN, X_trainML, y_trainNN, y_trainML = train_test_split(self.X_train, self.y_train, test_size=0.5)
        
        # seat leakage function
        leakyValue = 0
        leakyFunction = lambda leaky, percNN, percML: np.abs(percNN - percML) * leaky
        percNN = 0.5
        percML = 0.5

        for i in range(iterations):
            print('-----------------------------------------------------------------------------------------------')
            print(f'Iteration {i + 1}/{iterations}')
            print('-----------------------------------------------------------------------------------------------')

            if i != 0:
                #####################################################################################################
                # 2. (by-pass first loop) train a very simple MLM on to choose wich part of the data 
                #    to send to the neural network and wich to the BaggingClassifier
                #####################################################################################################
                print('Training the splitter model...')
                self.modelNNSplitter.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                # transform the target variable to a one-hot encoded vector
                y_train_NNvsML = tf.keras.utils.to_categorical(y_train_NNvsML, num_classes=2)
                self.modelNNSplitter.fit(X_trainMerge, y_train_NNvsML, epochs=10, batch_size=64)
                # calc accuracy
                y_predNNvsML = self.modelNNSplitter.predict(X_trainMerge)
                accuracy = np.mean(np.argmax(y_predNNvsML, axis=1) == y_train_NNvsML)
                print(f'Splitter model accuracy: {accuracy}')

                # set leaky function
                leakyValue = leakyFunction(leaky, percNN, percML)

                # split the data
                X_trainNN = X_trainMerge[np.where(np.argmax(y_predNNvsML, axis=1) == 0)]
                X_trainML = X_trainMerge[np.where(np.argmax(y_predNNvsML, axis=1) == 1)]
                y_trainNN = y_trainMerge[np.where(np.argmax(y_predNNvsML, axis=1) == 0)]
                y_trainML = y_trainMerge[np.where(np.argmax(y_predNNvsML, axis=1) == 1)]
                
                # percetage of the data to ML and NN
                percNN = len(X_trainNN) / len(X_trainMerge)
                percML = len(X_trainML) / len(X_trainMerge)
                print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
                print(f'% data to NN: {round(percNN * 100, 3)}%')
                print(f'% data to ML: {round(percML * 100, 3)}%')
                print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
                

            #####################################################################################################
            # 3. train the neural network with the first part
            #####################################################################################################
            print('Training the neural network...')
            self.modelNN.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            
            # leaky function correction
            if percML > percNN + leakyTolerance:
                print('Leaky function correction on NN...')
                print(f'Leaky value: {leakyValue}')
                randIdx = np.random.choice(len(X_trainNN), int(len(X_trainNN) * leakyValue))
                X_trainNN_tot = np.concatenate((X_trainNN, X_trainML[randIdx]))
                y_trainNN_tot = np.concatenate((y_trainNN, y_trainML[randIdx]))
            else:
                X_trainNN_tot = X_trainNN
                y_trainNN_tot = y_trainNN

            # transform the target variable to a one-hot encoded vector
            y_trainNN_tot = tf.keras.utils.to_categorical(y_trainNN_tot-1, num_classes=7)
            self.modelNN.fit(X_trainNN_tot, y_trainNN_tot, epochs=10, batch_size=64)
            # accuracy
            print(f'NN Accuracy: {self.modelNN.evaluate(X_trainNN_tot, y_trainNN_tot)[1]}')

            #####################################################################################################
            # 4. train the BaggingClassifier with the second part
            #####################################################################################################
            print('Training the BaggingClassifier...')

            # leaky function correction
            if percNN > percML + leakyTolerance:
                print('Leaky function correction on ML...')
                print(f'Leaky value: {leakyValue}')
                randIdx = np.random.choice(len(X_trainML), int(len(X_trainML) * leakyValue))
                X_trainML_tot = np.concatenate((X_trainML, X_trainNN[randIdx]))
                y_trainML_tot = np.concatenate((y_trainML, y_trainNN[randIdx]))
            else:
                X_trainML_tot = X_trainML
                y_trainML_tot = y_trainML
            
            self.modelML.fit(X_trainML_tot, y_trainML_tot)
            # accuracy
            print(f'ML Accuracy: {self.modelML.score(X_trainML, y_trainML)}')

            #####################################################################################################
            # 5. check wich part of the data is right and send the wrong part to the other model
            #####################################################################################################
            print('Checking the trained predictions...')
            y_predNN = self.modelNN.predict(X_trainNN)
            y_predML = self.modelML.predict(X_trainML)

            # check wich part of the data is right and send the wrong part to the other model
            # NN is 0 and ML is 1
            y_train_NNvsML = 2*np.ones(np.shape(self.y_train)[0])
            X_trainMerge = np.concatenate((X_trainNN, X_trainML))
            y_trainMerge = np.concatenate((y_trainNN, y_trainML))

            # check the predictions for the first part of the data NN
            for i in range(len(X_trainNN)):
                if np.argmax(y_predNN[i])+1 != y_trainNN[i]:
                    y_train_NNvsML[i] = 1
                else:
                    y_train_NNvsML[i] = 0

            # check the predictions for the second part of the data ML
            for i in range(len(X_trainML)):
                y_train_NNvsML[i + len(X_trainNN)] = 1
                if y_predML[i] != y_trainML[i]:
                    y_train_NNvsML[i + len(X_trainNN)] = 0

            # check if any data is not classified
            if 2 in y_train_NNvsML:
                print('Some data is not classified')
                exit()

    
    def predict(self, X_test):
        # prediction logic
        # X: input features
        # return: array of predictions
        # 1. predict the division of the data with the simple NN
        # 2. predict the first part of the data with the neural network
        # 3. predict the second part of the data with the BaggingClassifier
        # 4. merge the predictions

        #####################################################################################################
        # 1. predict the division of the data with the simple NN
        #####################################################################################################
        y_predNNvsML = self.modelNNSplitter.predict(X_test)

        # split the data
        indexNN = np.where(np.argmax(y_predNNvsML, axis=1) == 0)
        indexML = np.where(np.argmax(y_predNNvsML, axis=1) == 1)
        print(f'% data to NN: {round(len(indexNN[0]) / len(X_test) * 100, 3)}%')
        print(f'% data to ML: {round(len(indexML[0]) / len(X_test) * 100, 3)}%')
        X_testNN = X_test[indexNN]
        X_testML = X_test[indexML]

        #####################################################################################################
        # 2. predict the first part of the data with the neural network
        #####################################################################################################
        y_predNN = self.modelNN.predict(X_testNN)

        #####################################################################################################
        # 3. predict the second part of the data with the BaggingClassifier
        #####################################################################################################
        y_predML = self.modelML.predict(X_testML)

        #####################################################################################################
        # 4. merge the predictions
        #####################################################################################################
        y_pred = np.zeros(len(X_test))
        y_pred[indexNN] = np.argmax(y_predNN, axis=1)
        y_pred[indexML] = y_predML

        return y_pred, indexNN, indexML
    
    def evaluate(self, X_test, y_test):
        # evaluation logic
        # X: input features
        # y: input labels
        # return: accuracy
        print('\n\n\n')
        print('-----------------------------------------------------------------------------------------------')
        print('Evaluating the model...')
        print('-----------------------------------------------------------------------------------------------')
        y_pred, indexNN, indexML = self.predict(X_test)

        # accuracy
        accuracy = np.mean(y_pred == y_test)
        print(f'Accuracy: {accuracy}')

        # accuracy for each model
        accuracyNN = np.mean(y_pred[indexNN] == y_test[indexNN])
        print(f'NN Accuracy: {accuracyNN}')

        accuracyML = np.mean(y_pred[indexML] == y_test[indexML])
        print(f'ML Accuracy: {accuracyML}')

        return accuracy
