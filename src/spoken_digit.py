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
# GET TRAINING BATCH, returns data in batches 
#
def get_mfcc_batch(file_path, batch_size, utterance_length):
    print("hello")
    files = os.listdir(file_path)
    ft_batch = []
    label_batch = []

    while True:
        # Shuffle Files
        random.shuffle(files)
        for fname in files:
            # print("Total %d files in directory" % len(files))

            # Make sure file is a .wav file
            if not fname.endswith(".wav"):
                continue
            
            # Get MFCC Features for the file
            mfcc_features = extract_mfcc(file_path + fname, utterance_length)
            
            # One-hot encode label for 10 digits 0-9
            label = np.eye(10)[int(fname[0])]
            
            # Append to label batch
            label_batch.append(label)
            
            # Append mfcc features to ft_batch
            ft_batch.append(mfcc_features)

            # Check to see if default batch size is < than ft_batch
            if len(ft_batch) >= batch_size:
                # send over batch
                yield ft_batch, label_batch
                # reset batches
                ft_batch = []
                labels_batch = []

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
    iterations_train = 50
    bsize = 64
    audio_features = 20  
    utterance_length = 35  # Modify to see what different results you can get
    ndigits = 10

    # Get training data
    train_batch = get_mfcc_batch('../data/recordings/train/', 256, utterance_length)
    
    # # Build Model
    sp_network = tflearn.input_data([None, audio_features, utterance_length])
    sp_network = tflearn.lstm(sp_network, 128*4, dropout=0.5)
    sp_network = tflearn.fully_connected(sp_network, ndigits, activation='softmax')
    sp_network = tflearn.regression(sp_network, optimizer='adam', learning_rate=lr, loss='categorical_crossentropy')
    sp_model = tflearn.DNN(sp_network, tensorboard_verbose=0)

    # Train Model
    while iterations_train > 0:
        X_tr, y_tr = next(train_batch)
        X_test, y_test = next(train_batch)
        sp_model.fit(X_tr, y_tr, n_epoch=10, validation_set=(X_test, y_test), show_metric=True, batch_size=bsize)
        iterations_train -=1
    sp_model.save("../model/speech_recognition.lstm")

    # Test Model
    sp_model.load('../model/speech_recognition.lstm')
    mfcc_features = extract_mfcc('../data/recordings/test/3_jackson_49.wav', utterance_length)
    mfcc_features = mfcc_features.reshape((1,mfcc_features.shape[0],mfcc_features.shape[1]))
    prediction_digit = sp_model.predict(mfcc_features)
    print(prediction_digit)
    print("Digit predicted: ", np.argmax(prediction_digit))

    # Done
    return 0



if __name__ == '__main__':
    main()