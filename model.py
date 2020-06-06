from keras.models import Sequential, Model
from keras.layers import Input, Dense, Flatten, Lambda, GlobalAveragePooling2D, Cropping2D
from scipy import ndimage
from glob import glob
import numpy as np
import cv2
import csv
import math

# path_to_data = '/opt/carnd_p3/data/'
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
        for i in range(0, num_samples, batch_size//2):
            batch_samples = samples[i:i+batch_size//2]
            # read batch samples
            imgs = []
            measurements = []
            for sample in batch_samples:
                source_path = sample[0]
                filename = source_path.split('/')[-1]
                current_path = path_to_data + 'IMG/' + filename
                img = ndimage.imread(current_path)
                imgs.append(img)
                imgs.append(np.fliplr(img))
                measurements.extend([float(line[3]),-float(line[3])])
            X_batch = np.array(imgs)
            y_batch = np.array(measurements)
            yield shuffle(X_batch, y_batch)

BATCH_SIZE = 32
NUM_OF_EPOCHS = 5
train_generator = generator(train_samples, BATCH_SIZE)
valid_generator = generator(validation_samples, BATCH_SIZE)

print('building graph...')

def build_lenet():
    print('building lenet...')
    layer_input = Input(shape=(160, 320, 3))
    layer_flat = Flatten()(layer_input)
    layer_output = Dense(1)(layer_flat)
    model = Model(inputs=layer_input, outputs=layer_output)
    return model

def build_inception():
    print('building inception model...')
    from keras.applications.inception_v3 import InceptionV3
    inception = InceptionV3(include_top=False, weights='imagenet', input_tensor=None, input_shape=(160, 320, 3))
    for layer in inception.layers:
        layer.trainable = False
    layer_input = Input(shape=(160, 320, 3))
    print(layer_input.shape)
    layer_cropped = Cropping2D(cropping=((50,20), (0,0)))(layer_input)
    print(layer_cropped.shape)
    layer_norm = Lambda(lambda img: img/255.0 - 0.5)(layer_cropped)
    layer_inception = inception(layer_norm)
    layer_avg = GlobalAveragePooling2D()(layer_inception)
    layer_output = Dense(1)(layer_avg)
    model = Model(inputs=layer_input, outputs=layer_output)
    return model

print('training model...')
model = build_inception()
print(model.summary())
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit_generator(train_generator, \
                    steps_per_epoch=math.ceil(len(train_samples)/BATCH_SIZE), \
                    epochs=NUM_OF_EPOCHS, \
                    verbose=1, \
                    validation_data=valid_generator, \
                    validation_steps=math.ceil(len(validation_samples)/BATCH_SIZE))
model.save('model.h5')
