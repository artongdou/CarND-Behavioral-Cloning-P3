import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import math

# Import for workspace
# from keras import Model, Sequential
# from keras.layers import Lambda, MaxPooling2D, Dropout, Flatten, Dense, Conv2D, Input, Cropping2D
# from keras.callbacks import EarlyStopping
# from keras import backend

# import for local PC
from tensorflow_core.python.keras import Model, Sequential
from tensorflow_core.python.keras.api._v2.keras.layers import Lambda, MaxPooling2D, Dropout, Flatten, Dense, Conv2D, Input, Cropping2D
from tensorflow_core.python.keras.callbacks import EarlyStopping
from tensorflow_core.python.keras import backend

import tensorflow as tf
from sklearn.utils import shuffle

backend.clear_session()

# mydata_path = '/opt/carnd_p3/data'
mydata_path = '../run4'
img_rows = 66
img_cols = 200

# load logs
logs = []
with open(mydata_path + '/driving_log.csv','rt') as f:
    reader = csv.reader(f)
    for line in reader:
        # print(line)
        logs.append(line)
    # logs.pop(0)
print("number of images: ", len(logs))

# samples
samples = []
labels = []
steering_offset = 0.25

def preprocess_img(img):
    '''
    @param an image array in BGR format
    '''
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = img[60:140, 0:320]
    img = cv2.resize(img, (img_cols, img_rows)) # resize expected (width, height)
    _,s,_ = cv2.split(img)
    s = np.reshape(s, (img_rows, img_cols, 1))
    return s

def random_shift(img):
    offset = math.floor(random.uniform(50, 150))
    M_right = np.float32([[1, 0, offset], [0, 1, 0]]) 
    M_left = np.float32([[1, 0, -offset], [0, 1, 0]]) 
    (rows, cols) = img.shape[:2] 
  
    # warpAffine does appropriate shifting given the 
    # translation matrix. 
    shifted_right_img = cv2.warpAffine(img, M_right, (cols, rows)) 
    shifted_left_img = cv2.warpAffine(img, M_left, (cols, rows)) 
    shifted_right_img = np.reshape(shifted_right_img, (rows, cols, 1))
    shifted_left_img = np.reshape(shifted_left_img, (rows, cols, 1))
    return shifted_left_img, shifted_right_img, offset

def load_data():
    #load center -> left -> right images
    pixel_to_angle = 0.01
    for i in range(3):
        for j in range(len(logs)):
        # for j in range(1000):
            if float(logs[j][3]) <= 0.001:
                continue
            img_filename = logs[j][i].split('\\')[-1].strip()
            # print(img_filename)
            img_path = mydata_path + '/IMG/' + img_filename
            # print(img_path)
            img = cv2.imread(img_path)
            pp_img = preprocess_img(img)
            samples.append(pp_img)
            samples.append(np.fliplr(pp_img))
            l, r, pixel_shifted = random_shift(pp_img)
            samples.append(l)
            samples.append(r)
            flip_l, flip_r, flip_pixel_shifted = random_shift(np.fliplr(pp_img))
            samples.append(flip_l)
            samples.append(flip_r)
            if i == 0:
                # center image
                # labels.append(float(logs[j][3]))
                # labels.append(-float(logs[j][3]))
                steering = float(logs[j][3])
            elif i == 1:
                # left image
                # labels.append(float(logs[j][3]) + steering_offset)
                # labels.append(-float(logs[j][3]) - steering_offset)
                steering = float(logs[j][3]) + steering_offset
            else:
                # right image
                # labels.append(float(logs[j][3]) - steering_offset) 
                # labels.append(-float(logs[j][3]) + steering_offset)
                steering = float(logs[j][3]) - steering_offset
            labels.extend([steering, -steering, 
                           steering + pixel_shifted*pixel_to_angle, steering - pixel_shifted*pixel_to_angle,
                           -steering + pixel_shifted*pixel_to_angle, -steering - pixel_shifted*pixel_to_angle])

def test():
    imgc = cv2.imread("C:\\Users\\Chishing\\Desktop\\beta_simulator_windows\\run2\\IMG\\left_2020_06_07_02_27_11_352.jpg")
    imgl = cv2.imread("C:\\Users\\Chishing\\Desktop\\beta_simulator_windows\\run2\\IMG\\center_2020_06_07_02_27_11_352.jpg")
    imgr = cv2.imread("C:\\Users\\Chishing\\Desktop\\beta_simulator_windows\\run2\\IMG\\right_2020_06_07_02_27_11_352.jpg")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle('Horizontally stacked subplots')
    ax1.imshow(np.reshape((preprocess_img(imgl)), (img_rows, img_cols)), cmap='gray')
    ax2.imshow(np.reshape((preprocess_img(imgc)), (img_rows, img_cols)), cmap='gray')
    ax3.imshow(np.reshape((preprocess_img(imgr)), (img_rows, img_cols)), cmap='gray')
    plt.show()

def simple_model():
    model = Sequential([
                    Lambda(lambda x: x/255 - 0.5, input_shape=(img_rows,img_cols,1)),
                    Cropping2D(cropping=((math.floor(img_rows*3/8),math.floor(img_rows/8)),(0,0))),
                    Conv2D(filters=16, kernel_size=5, strides=1, padding='valid', activation='relu'),
                    MaxPooling2D((4,4),(4,4),'valid'),
                    Dropout(0.5),
                    Flatten(),
                    Dense(1)
                ])
    return model

def nvidia():
    return Sequential([
        Lambda(lambda x: x/255 - 0.5, input_shape=(img_rows,img_cols,1)),
        Conv2D(24, 5, 2, padding='valid', activation='elu'),
        Conv2D(26, 5, 2, padding='valid', activation='elu'),
        Conv2D(48, 5, 2, padding='valid', activation='elu'),
        Conv2D(64, 3, 1, padding='valid', activation='elu'),
        Conv2D(64, 3, 1, padding='valid', activation='elu'),
        Flatten(),
        Dropout(0.5),
        Dense(100),
        Dropout(0.5),
        Dense(50),
        Dropout(0.5),
        Dense(10),
        Dropout(0.5),
        Dense(1)
    ])

if __name__ == '__main__':
    # pass
    load_data()
    print(samples[0].shape)
    X = np.array(samples)
    print("samples shape: {}".format(X.shape))
    y = np.array(labels)
    print("labels shape: {}".format(y.shape));
    X, y = shuffle(X, y)

    # plt.hist(y, bins=19)
    # plt.show()

    model = simple_model()
    # model = nvidia()
    model.summary()
    model.compile(optimizer='adam', loss='mean_squared_error')

    early_stop_cb = EarlyStopping(monitor='loss', patience=3)
    model.fit(X, y, batch_size=512, epochs=15, verbose=1,
                validation_split=0.2, shuffle=True, callbacks=[early_stop_cb])
    model.save('model.h5')