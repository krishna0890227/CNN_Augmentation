from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import cluster as cl
import tensorflow as tf
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D, AveragePooling1D
from keras.layers.advanced_activations import LeakyReLU, PReLU
import numpy as np
########################################################################


def MLP_learning(X_train, y_train, X_test, y_test, baseline):
    a=X_train.shape[1]

    #######################################################################################
    model = Sequential()
    model.add(Dense(32, input_dim=a, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(32, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(32, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='linear'))
    # # compile the model
    model.compile(loss="mean_squared_error", optimizer="rmsprop",metrics=['accuracy'])
    model.summary()
    # # train the model
    model.fit(X_train, y_train, batch_size=32, nb_epoch=150, validation_split=0.10, verbose=0)
    ANN_pred1 = model.predict(X_test)
    print('close here.............')
    type(ANN_pred1)
    ANN_pred2=np.array(ANN_pred1)
    ANN_pred3=ANN_pred2.flatten()
    ANN_pred=ANN_pred3+ baseline
    ANN_rmse=cl.RMse(ANN_pred, y_test)
    ANN_mape=cl.MApe(ANN_pred, y_test)
    print(ANN_rmse, ANN_mape)
    print(y_test)
    print(ANN_pred)
    return ANN_pred, ANN_mape, ANN_rmse


def CNN_network(X_train_3D, y_train, X_test_3D, y_test, baseline):
    a=X_train_3D.shape[1]
    model=Sequential()
    model.add(Conv1D(input_shape=(a,1), nb_filter=12, filter_length=3, border_mode='valid', activation='relu'))
    model.add(MaxPooling1D(pool_length=3))
    model.add(Conv1D(input_shape=(a,1), nb_filter=12, filter_length=3, border_mode='valid', activation='relu'))
    #model.add(Conv1D(input_shape=(a, 1), nb_filter=12, filter_length=3, border_mode='valid', activation='relu'))
    #model.add(Conv1D(input_shape=(a, 1), nb_filter=12, filter_length=3, border_mode='valid', activation='relu'))
    # model.add(Conv1D(input_shape=(a, 1), nb_filter=12, filter_length=3, border_mode='valid', activation='relu'))
    # model.add(Conv1D(input_shape=(a, 1), nb_filter=12, filter_length=3, border_mode='valid', activation='relu'))
   # #  model.add(Conv1D(input_shape=(a, 1), nb_filter=12, filter_length=3, border_mode='valid', activation='relu'))
   #  model.add(Conv1D(input_shape=(a, 1), nb_filter=12, filter_length=3, border_mode='valid', activation='relu'))
    #model.add(AveragePooling1D(pool_length=3))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))
    model.summary()
    model.compile(loss='mean_squared_error',optimizer="rmsprop",metrics=['accuracy'])
    ### train the model
    model.fit(X_train_3D, y_train, batch_size=32, nb_epoch=150, validation_split=0.15, verbose=0)
    CNN_pred1=model.predict(X_test_3D)
    CNN_pred2 = np.array(CNN_pred1)
    CNN_pred3 = CNN_pred2.flatten()
    CNN_pred=CNN_pred3 + baseline
    CNN_rmse=cl.RMse(CNN_pred, y_test)
    CNN_mape=cl.MApe(CNN_pred, y_test)
    print(CNN_rmse, CNN_mape)
    print(y_test)
    return CNN_pred, CNN_rmse, CNN_mape

def LSTM_network(X_train_3D, y_train, X_test_3D, y_test, baseline):
    model=Sequential()
    model.add(LSTM(23, input_dim=1, return_sequences=True, kernel_initializer='uniform', activation='tanh'))
    model.add(Dropout(0.5))
    model.add(LSTM(12,kernel_initializer='uniform', return_sequences=False,  activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))
    model.compile (loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
    model.summary()
    model.fit(X_train_3D, y_train, batch_size=32, nb_epoch=300, validation_split=0.15, verbose=0)
    LSTM_pred1=model.predict(X_test_3D)
    LSTM_pred2 = np.array(LSTM_pred1)
    LSTM_pred3 = LSTM_pred2.flatten()
    LSTM_pred=LSTM_pred3 + baseline
    LSTM_rmse=cl.RMse(LSTM_pred, y_test)
    LSTM_mape=cl.MApe(LSTM_pred, y_test)
    print(LSTM_rmse, LSTM_mape)
    print(LSTM_pred)
    return LSTM_pred, LSTM_rmse, LSTM_mape

def dense_pooled(X_train, y_train, X_test,y_test, shifted_value):
    IS = X_train.shape[1]
    model = Sequential()
    model.add(Dense(32, input_dim=IS, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(0.5))
    # model.add(Dense(100, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(32, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))
    model.summary()
    # # compile the model
    model.compile(loss="mean_squared_error", optimizer="rmsprop", metrics=['accuracy'])
    # # train the model
    model.fit(X_train, y_train, batch_size=32, nb_epoch=150, validation_split=0.10, verbose=0)
    ######################################################################################
    ANN_pred = model.predict(X_test)
    print(ANN_pred.shape)
    y_ANN = ANN_pred + shifted_value
    print(y_ANN.shape)
    #print(y_ANN)
    ANN_mape=cl.RMse(y_test,y_ANN)
    ANN_rmse=cl.MApe(y_test, y_ANN)
    ANN_mape=np.array(ANN_mape)
    ANN_rmse=np.array(ANN_rmse)
    print(ANN_mape, ANN_rmse)
    print(y_test)
    #print(np.array(ANN_pred))
    return y_ANN, ANN_rmse, ANN_mape

def CNN_net(X_train_3D, y_train, X_test_3D,y_test, shifted_value):
    a = X_train_3D.shape[1]
    model3 = Sequential()
    model3.add(Conv1D(input_shape=(a, 1), nb_filter=12, filter_length=3, border_mode='valid', activation='linear',
                      subsample_length=1))
    model3.add(LeakyReLU(alpha=0.001))
    model3.add(MaxPooling1D(pool_length=3))
    model3.add(Dropout(0.2))
    model3.add(Conv1D(input_shape=(a, 1), nb_filter=12, filter_length=3, border_mode='valid', activation='linear',
                      subsample_length=1))
    model3.add(LeakyReLU(alpha=0.001))
    model3.add(Dropout(0.2))
    # model3.add(Conv1D(input_shape=(a, 1), nb_filter=29, filter_length=3, border_mode='valid', activation='linear',
    #                subsample_length=1))
    # model3.add(LeakyReLU(alpha=0.001))
    # model3.add(Dropout(0.25))
    model3.add(Flatten())

    model3.add(Dense(128, activation='linear'))
    model3.add(LeakyReLU(alpha=0.001))
    # model3.add(Dense(128, activation='linear'))
    # model3.add(LeakyReLU(alpha=0.001))
    # model3.add(Dropout(0.5))
    model3.add(Dense(1))
    model3.add(Activation('linear'))
    model3.summary()

    model3.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
    model3.fit(X_train_3D, y_train, batch_size=32, validation_split=0.10, epochs=150, verbose=0)
    y_CNN = model3.predict(X_test_3D)
    CNN_pred = y_CNN +shifted_value
    CNN_mape = cl.MApe(y_test, CNN_pred)
    CNN_rmse = cl.RMse(y_test, CNN_pred)
    CNN_mape = np.array(CNN_mape)
    CNN_rmse = np.array(CNN_rmse)
    print(CNN_mape, CNN_rmse)
    print(y_test)
    #print(CNN_pred)
    return CNN_pred, CNN_rmse, CNN_mape

def LSTM_net(X_train_3D, y_train, X_test_3D, y_test, shifted_value):
    model1 = Sequential()
    model1.add(LSTM(20, input_dim=1, return_sequences=True, kernel_initializer='uniform', activation='tanh',
                    inner_activation='hard_sigmoid'))
    model1.add(Dropout(0.5))
    model1.add(LSTM(20, kernel_initializer='uniform', return_sequences=False, activation='tanh',
                     inner_activation='hard_sigmoid'))
    # model1.add(Dropout(0.5))
    # model1.add(LSTM(64, kernel_initializer='uniform', return_sequences=False, activation='tanh',
    #                 inner_activation='hard_sigmoid'))
    model1.add(Dropout(0.5))
    model1.add(Dense(1, activation='linear'))
    # # compile the model
    model1.compile(loss="mean_squared_error", optimizer="rmsprop", metrics=['accuracy'])
    # # train the model
    model1.fit(X_train_3D, y_train, batch_size=32, nb_epoch=300, validation_split=0.10, verbose=0)
    model1.summary()

    LSTM_pred = model1.predict(X_test_3D)
    y_LSTM = LSTM_pred + shifted_value
    # y_GRU=scalery.inverse_transform(GRU_pred)
    LSTM_mape = cl.MApe(y_test, y_LSTM)
    LSTM_rmse = cl.RMse(y_test, y_LSTM)
    LSTM_mape = np.array(LSTM_mape)
    LSTM_rmse = np.array(LSTM_rmse)
    print(LSTM_mape, LSTM_rmse)
    print(y_test)
    #print(np.array(y_LSTM))
    return y_LSTM, LSTM_rmse, LSTM_mape
