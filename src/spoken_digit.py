# -*- coding: utf-8 -*-
import os
import time
import random
import tflearn
import librosa
import numpy as np
import librosa.display
import tensorflow as tf
import matplotlib.pyplot as plt

#
# EXTRACT MFCC FEATURES
#
def extract_mfcc(file_path, utterance_length):
    # Get raw .wav data and sampling rate from librosa's load function
    raw_w, sampling_rate = librosa.load(file_path, mono=True)

    # Obtain MFCC Features from raw data
    mfcc_features = librosa.feature.mfcc(raw_w, sampling_rate)
    if mfcc_features.shape[1] > utterance_length:
        mfcc_features = mfcc_features[:, 0:utterance_length]
    else:
        mfcc_features = np.pad(mfcc_features, ((0, 0), (0, utterance_length - mfcc_features.shape[1])),
                               mode='constant', constant_values=0)
    
    return mfcc_features

#
# GET MFCC BATCH
#

#
# DISPLAY FEATURE SHAPE
#
# wav_file_path: Input a file path to a .wav file
#
def display_power_spectrum(wav_file_path, utterance_length):
    mfcc = extract_mfcc(wav_file_path, utterance_length)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    librosa.display.specshow(mfcc, x_axis='time')
    plt.show()

    # Feature information
    print("Feature Shape: ", mfcc.shape)
    print("Features: " , mfcc[:,0])

#
# MAIN
#
def main():
    # Initial Parameters
    lr = 0.001
    iterations_train = 30
    bsize = 64
    audio_features = 20  
    utterance_length = 35  # Modify to see what different results you can get
    ndigits = 10

    # Test
    test_file = '../data/recordings/train/0_jackson_13.wav'
    display_power_spectrum(test_file, utterance_length)

    # Preprocess Data

    # Build Model

    # Train Model

    # Test Model
    
     
    # Done
    print("DONE")
    return 0



if __name__ == '__main__':
    main()