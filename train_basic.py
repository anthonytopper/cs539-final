import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, TimeDistributed, Activation # CuDNNLSTM
from tensorflow.keras import backend as K
from tempfile import TemporaryFile

import load_sample



def zeropad(input,size):
    x = input['x']
    y = input['y']
    result = {}
    result['x'] = np.pad(x,((0,size-len(x)),(0,0)),'constant')
    result['y'] = np.pad(y,((0,size-len(y)),(0,0)),'constant')
    return result

samples_train = [load_sample.load(s) for s in SAMPLES_TRAIN]
samples_test = [load_sample.load(s) for s in SAMPLES_TEST]

# Maximum size for all samples
max_size = max([len(s['x']) for s in (samples_train+samples_test)])

samples_train = [zeropad(s,max_size) for s in samples_train]
samples_test = [zeropad(s,max_size) for s in samples_test]


tf.reset_default_graph()
K.clear_session()

#np.array([[next(([i] for i, x in enumerate(m) if x), None) for m in midi_samples[20000:23657]]])

# [samples, time steps, features]

x_train = np.array([s['x'] for s in samples_train])
y_train = np.array([s['y'] for s in samples_train])

x_test = np.array([s['x'] for s in samples_test])
y_test = np.array([s['y'] for s in samples_test])
# y_train = np.array([[tf.keras.utils.to_categorical(np.argmax(m),30) for m in midi_samples[0:n_input_train]]])#np.array([[next(([i] for i, x in enumerate(m) if x), None) for m in midi_samples[0:20000]]])

# x_test = np.array([freq_boxes[n_input_train:n_total]])
# y_test = np.array([[tf.keras.utils.to_categorical(np.argmax(m),30) for m in midi_samples[n_input_train:n_total]]])


input_shape = (None,x_train.shape[2])

model = Sequential()

# TODO: switch to CuDNNLSTM on g4 EC2
model.add(LSTM(255, input_shape=input_shape, return_sequences=True)) # CuDNNLSTM
model.add(LSTM(30, input_shape=input_shape, return_sequences=True))
# model.add(Dropout(0.2))
# model.add(Activation('relu'))
model.add(TimeDistributed(Dense(30, activation='softmax')))
# model.add(Dropout(0.2))


opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
print(model.summary()) 
model.compile(
    loss='categorical_crossentropy',
    optimizer=opt,
    metrics=['categorical_accuracy'],
)


print('done compiling')
print(model.fit(x_train,
          y_train,
          verbose=2,
          epochs=5))

print('done fitting')
print(model.evaluate(x_test,y_test))
print('done evaluating')

model.save(MODEL_ID+'.h5')