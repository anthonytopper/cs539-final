import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, TimeDistributed, Activation, Flatten, GRU # CuDNNLSTM
from tensorflow.keras import backend as K
from tempfile import TemporaryFile
import pandas as pd
import random
import load_sample

process = True

if process:
    def zeropad(input,size):
        x = input['x']
        y = input['y']
        result = {}
        result['x'] = np.pad(x,((0,size-len(x)),(0,0)),'constant')
        result['y'] = y

        return result
    file_list = ['AudioWAV/%s' % f for f in os.listdir('AudioWAV') if os.path.isfile('AudioWAV/%s' % f)][:10]
    file_list = file_list #[0:200]
    
    file_list = [load_sample.load(s) for s in file_list]
    file_list = [s for s in file_list if s]
    max_size = max([len(s['x']) for s in (file_list)])

    file_list = [zeropad(s,max_size) for s in file_list]

    random.shuffle(file_list)
    
    samples_train = file_list[:int(len(file_list) * 0.7)]
    samples_test  = file_list[int(len(file_list) * 0.7):]
    
    x_train = np.array([s['x'] for s in samples_train])
    y_train = np.array([s['y'] for s in samples_train])

    x_test = np.array([s['x'] for s in samples_test])
    y_test = np.array([s['y'] for s in samples_test])

    np.save('x_train', x_train)
    np.save('y_train', y_train)
    np.save('x_test', x_test)
    np.save('y_test', y_test)

else:
    x_train = np.load('x_train.npy')
    y_train = np.load('y_train.npy')
    x_test = np.load('x_test.npy')
    y_test = np.load('y_test.npy')

input_shape = (x_train.shape[1:])

model = Sequential()
model.add(GRU(90, return_sequences=True,dropout=0.2, input_shape=(x_train.shape[1],4)))
model.add(GRU(60,dropout=0.2, return_sequences=True))
model.add(Flatten())
model.add(Dense(60))
model.add(Dropout(0.2))
model.add(Dense(30))
model.add(Dropout(0.2))
model.add(Dense(6, activation='softmax'))

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

print(model.summary()) 
model.compile(
    loss='categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy'],
)

print('done compiling')
print(model.fit(x_train,
          y_train,
          verbose=2,
          epochs=100))

print('done fitting')
print(model.evaluate(x_test,y_test))
print('done evaluating')

# model.save(MODEL_ID+'.h5')
