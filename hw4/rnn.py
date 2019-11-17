import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import math

def processData(look_back):    
    # Load the data
    data = np.load('data.npy')
    data = data.astype('float32')
    
    # Normalize and scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data) 
    
    # Split Train and test data
    train_data = data[:66]
    test_data  = data[66:]
    x_train, y_train = createXY(train_data, look_back)
    x_test, y_test   = createXY(test_data, look_back)
    return x_train, y_train, x_test, y_test

def createXY(data, look_back=1):
    x, y = [], []
    for i in range(len(data)-look_back-1):            
        tempData = data[i:(i+look_back), 0]
        x.append(tempData)
        y.append(data[i + look_back, 0])
	
    return np.array(x), np.array(y)

def createModel(look_back):    
    # create the LSTM network
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(10, batch_input_shape=(None, 1, look_back), return_sequences=True))
    model.add(tf.keras.layers.LSTM(20, return_sequences=True))
    model.add(tf.keras.layers.LSTM(20))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    model.summary()
    return model

look_back = 10
epochs = 100
batch_size = 10

(x_train, y_train, x_test, y_test) = processData(look_back)

# reshape input to be [samples, time steps, features]
x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
x_test =  np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

model = createModel(look_back)

model.compile(loss='mean_squared_error', optimizer='adam')

# train the model
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

# Make Prediction
y_train_predict = model.predict(x_train)
y_test_predict = model.predict(x_test)

# reshape to match predicted values array
y_train = np.reshape(y_train, (y_train.shape[0], 1))
y_test = np.reshape(y_test, (y_test.shape[0], 1))

# calculate root mean squared error
trainRMSE = math.sqrt(mean_squared_error(y_train[0], y_train_predict[0]))
print(f'Training RMSE: {trainRMSE}\n')
testRMSE = math.sqrt(mean_squared_error(y_test[0], y_test_predict[0]))
print(f'Test RMSE: {testRMSE}\n')