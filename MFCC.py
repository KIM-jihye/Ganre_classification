import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank    #로그 필터 뱅크 에너지

# Read input sound file
sampling_freq, audio = wavfile.read("csong.wav")

# MFCC 필터 뱅크 기능 추출
mfcc_features = mfcc(audio, sampling_freq *20)
filterbank_features = logfbank(audio, sampling_freq *20)

# Print parameters
print ('\nMFCC:\nNumber of windows =', mfcc_features.shape[0])
print ('Length of each feature =', mfcc_features.shape[1])
print ('\nFilter bank:\nNumber of windows =', filterbank_features.shape[0])
print ('Length of each feature =', filterbank_features.shape[1])

# Plot the features
mfcc_features = mfcc_features.T
plt.matshow(mfcc_features)
plt.title('MFCC')

#filterbank_features = filterbank_features.T
plt.matshow(filterbank_features)
plt.title('Filter bank')
plt.show()