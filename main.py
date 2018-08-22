import sys
from features import make_features
import scipy.io.wavfile as wav
import numpy as np
from keras.models import model_from_json

folder_name = "5second";

# read audio file and calculate mfcc
audio_file_name = sys.argv[1]
(rate, sig) = wav.read("1.wav")
mfcc_feat = make_features(sig, rate, winlen=0.1, winstep=0.05, lowfreq=50, highfreq=5000)
mfcc_feat[~np.isnan(mfcc_feat)]

# create feature
img_rows, img_cols = 401, 13
X_test = np.array([], dtype='float32')
mfcc_feat = mfcc_feat[0:401, :]
image = np.array([mfcc_feat])
if len(image) == 1:
    if len(X_test) == 0:
        X_test = np.array([image]);
    else:
        X_test = np.vstack([X_test, np.array([image])])

X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_test = X_test.astype('float32')
X_test /= 255

# load model
f = open('%s_model.json' % folder_name, 'r')
json_string = f.read()
model = model_from_json(json_string)

# load weight
model.load_weights('%s_weight.h5' % folder_name)
model.compile(loss='categorical_crossentropy', optimizer='adadelta')

# test dataset
genres = ["csong", "hiphop", "jazz", "balad", "edm", "ratin"];  # blues, kpop, edm, "metal"
result = model.predict_classes(X_test)
#print genres[result()]


