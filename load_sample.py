import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.io import wavfile
import json 
import tensorflow as tf
import math

labels = ['SAD','HAP'] # 'ANG'

from preprocess_parselmouth import preproc_intensity

def load(filename):

    y = get_label(filename)

    if y is None:
        print('Skipping %s' % filename)
        return

    rate, data = wavfile.read(filename) # load the data
    normalized = [(e / 2 ** 8.) * 2 - 1 for e in data] # normalized on [-1,1)

    freq_boxes = []

    sample_step = 50
    sample_rate = 16000
    NFFT = 1024
    duration = len(normalized) / rate # seconds
    window = np.hanning(NFFT)

    for i in range(0,len(normalized)-NFFT,sample_step):
        block = normalized[i:i+NFFT]
        block = np.multiply(block,window)
        freq = fft(block)
        freq = freq[:(math.floor(len(freq)/2))]
        freq = np.real(freq)
        freq_boxes.append(freq)
        

    # plt.specgram(normalized,NFFT=256, Fs=2, Fc=0)
    # plt.savefig(sample_id+'-specgram.png')

    x = freq_boxes

    x = np.array(preproc_intensity(data,rate)).T

    print('Done parsing %s with %d samples of size %d each' % (filename,len(x),len(x[0])))

    return {'x':x,'y':y}

def get_label(filename):
    for i,l in enumerate(labels):
        if l in filename:
            return tf.one_hot([i], len(labels))

    return None