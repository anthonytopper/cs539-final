import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.io import wavfile
import json 
import tensorflow as tf
import math

def load(sample_id):

    rate, data = wavfile.read(sample_id+'.wav') # load the data
    normalized = [(e / 2 ** 8.) * 2 - 1 for e in data] # normalized on [-1,1)

    freq_boxes = []

    sample_step = 10
    sample_rate = 8000
    NFFT = 512
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

    return freq_boxes

