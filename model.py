from keras.models import Sequential, Model
from keras.layers import Input, Dense, Flatten
from scipy import ndimage
from glob import glob
import numpy as np
import cv2
import csv
import math

path_to_data = '/opt/mydata/data/'

print('reading csv log file...')
samples = []
with open(path_to_data+'driving_log.csv') as f:
    reader = csv.reader(f)
    for line in reader:
        samples.append(line)
del(samples[0]) # delete the header line

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print('defining generator...')
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while True:
        shuffle(samples)
        for i in range(0, num_samples, batch_size):
            batch_samples = samples[i:i+batch_size]
            # read batch samples
            imgs = []
            measurements = []
            for sample in samples:
                source_path = sample[0]
                filename = source_path.split('/')[-1]
                current_path = path_to_data + 'IMG/' + filename
                img = ndimage.imread(current_path)
                imgs.append(img)
                measurements.append(float(line[3]))
            X_batch = np.array(imgs)
            y_batch = np.array(measurements)
            yield shuffle(X_batch, y_batch)

BATCH_SIZE = 32
train_generator = generator(train_samples, BATCH_SIZE)
valid_generator = generator(validation_samples, BATCH_SIZE)

print('building graph...')
layer_input = Input(shape=(160, 320, 3))
layer_flat = Flatten()(layer_input)
layer_output = Dense(1)(layer_flat)
model = Model(inputs=layer_input, outputs=layer_output)

print('training model...')
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit_generator(train_generator, \
                    steps_per_epoch=math.ceil(len(train_samples)/BATCH_SIZE), \
                    epochs=5, \
                    verbose=1, \
                    validation_data=valid_generator, \
                    validation_steps=math.ceil(len(validation_samples)/BATCH_SIZE))
