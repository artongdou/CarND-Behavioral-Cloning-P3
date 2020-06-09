import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import math
import datetime
import os
from sklearn.utils import shuffle

# Import for workspace
# from keras import Model, Sequential
# from keras.layers import Lambda, MaxPooling2D, Dropout, Flatten, Dense, Conv2D, Input, Cropping2D
# from keras.callbacks import EarlyStopping, TensorBoard
# from keras import backend
# from keras.optimizers import Adam

# import for local PC
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Lambda, MaxPooling2D, Dropout, Flatten, Dense, Conv2D, Input, Cropping2D
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras import backend
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam

backend.clear_session()

# mydata_path = '/opt/carnd_p3/data'
# mydata_path = '../run4'
img_rows = 16
img_cols = 32
BATCH_SIZE = 32
EPOCHS = 15
crop_top = 60
crop_bottom = 20
camera_steering_offset = 0.25

# Dict to desribe csv logfile columns
csv_cols = {'center': 0, 'left': 1, 'right': 2, 'steering': 3}

# Dict to store features and labels
data = {'feature': [], 'label': []}

def preprocess_img(img):
    '''
    @param an image array in BGR format
    '''

    # Convert to HSV colorspace
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    rows, cols, _ = img.shape

    # Crop top and bottom of the image
    img = img[0 + crop_top: rows - crop_bottom, 0:cols]
    # plt.imshow(img)
    # plt.show()

    # Resize to network input size
    img = cv2.resize(img, (img_cols, img_rows)) # resize expected (width, height)

    # Return only "Saturation" channel
    _,s,_ = cv2.split(img)

    # Reshape to 3D arrays
    s = np.reshape(s, (img_rows, img_cols, 1))

    return s

def random_shift(img):
    offset = math.floor(random.uniform(5, 15))
    M_right = np.float32([[1, 0, offset], [0, 1, 0]]) 
    M_left = np.float32([[1, 0, -offset], [0, 1, 0]]) 
    rows, cols, _ = img.shape 
  
    # warpAffine does appropriate shifting given the 
    # translation matrix. 
    shifted_right_img = cv2.warpAffine(img, M_right, (cols, rows)) 
    shifted_left_img = cv2.warpAffine(img, M_left, (cols, rows))

    # reshape to 3D arrays
    shifted_right_img = np.reshape(shifted_right_img, (rows, cols, 1))
    shifted_left_img = np.reshape(shifted_left_img, (rows, cols, 1))

    return shifted_left_img, shifted_right_img, offset

def visualize_data():
    steering = [np.float32(logs[i][3]) for i in range(len(logs))]
    plt.hist(steering, bins=19)
    plt.show()

def load_data(data_path, visualize = False, pixel_to_angle = 0.01):
    # load log file
    logs = []
    with open(data_path + '/driving_log.csv','rt') as f:
        reader = csv.reader(f)
        for line in reader:
            logs.append(line)
    print("number of images: ", len(logs))
    if visualize:
        visualize_data()

    #load center -> left -> right images
    for i in range(3):
        for j in range(len(logs)):
            steering = float(logs[j][csv_cols['steering']])
            if math.fabs(steering) <= 0.001:
                continue
            img_filename = logs[j][i].split('\\')[-1].strip()
            img_path = data_path + '/IMG/' + img_filename
            img = cv2.imread(img_path)

            # Pre-process image
            pp_img = preprocess_img(img)

            # Augment image
            l, r, pixel_shifted = random_shift(pp_img)
            flip_l, flip_r, flip_pixel_shifted = random_shift(np.fliplr(pp_img))
            
            # Calculate steering compensation for augmented image
            if i == csv_cols['center']:
                # center image
                pass
            elif i == csv_cols['left']:
                # left image
                steering += camera_steering_offset
            elif i == csv_cols['right']:
                # right image
                steering -= camera_steering_offset
            
            # Append feature and corresponding labels
            data['feature'].append(pp_img)
            data['feature'].append(np.fliplr(pp_img))
            data['feature'].append(l)
            data['feature'].append(r)
            data['feature'].append(flip_l)
            data['feature'].append(flip_r)
            data['label'].extend([steering, -steering, 
                           steering + pixel_shifted*pixel_to_angle, steering - pixel_shifted*pixel_to_angle,
                           -steering + flip_pixel_shifted*pixel_to_angle, -steering - flip_pixel_shifted*pixel_to_angle])

# def test():
#     imgc = cv2.imread("../run4_curve/IMG/left_2020_06_08_17_54_15_939.jpg")
#     imgl = cv2.imread("../run4_curve/IMG/center_2020_06_08_17_54_15_939.jpg")
#     imgr = cv2.imread("../run4_curve/IMG/right_2020_06_08_17_54_15_939.jpg")
#     fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
#     fig.suptitle('Horizontally stacked subplots')
#     ax1.imshow(np.reshape((preprocess_img(imgl)), (img_rows, img_cols)), cmap='gray')
#     ax2.imshow(np.reshape((preprocess_img(imgc)), (img_rows, img_cols)), cmap='gray')
#     ax3.imshow(np.reshape((preprocess_img(imgr)), (img_rows, img_cols)), cmap='gray')
#     plt.show()

def simple_model():
    model = Sequential([
                    Lambda(lambda x: x/127.5 - 1, input_shape=(img_rows,img_cols,1)),
                    Conv2D(filters=16, kernel_size=3, strides=1, padding='valid', activation='elu'),
                    MaxPooling2D((2,2),(2,2),'valid'),
                    Conv2D(filters=32, kernel_size=3, strides=1, padding='valid', activation='elu'),
                    MaxPooling2D((2,2),(2,2),'valid'),
                    Dropout(0.5),
                    Flatten(),
                    Dense(100, activation='elu'),
                    Dropout(0.5),
                    Dense(1)
                ])

    # learning rate decay (Only work in tf2.x)
    # lr_schedule = ExponentialDecay(
    #     initial_learning_rate=0.001,
    #     decay_steps=10000,
    #     decay_rate=0.9
    # )
    # optmzr = Adam(learning_rate=lr_schedule
    optmzr = Adam(lr=0.001, decay=0.1)

    # Compile graph
    model.compile(optimizer=optmzr, loss='mean_squared_error')

    return model

def get_callbacks(early_stop = False, tensorboard_log = False):
    callbacks = []
    # Early stop callback
    if early_stop:
        callbacks.append(EarlyStopping(monitor='loss', patience=3))

    if tensorboard_log:
        # Create log directory
        if not os.path.exists('log'):
            os.mkdir('log')
        if not os.path.exists('log/fit'):
            os.mkdir('log/fit')
        
        # create folder name using timestamp
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.normpath(log_dir)
        os.mkdir(log_dir)
        print("log directory: ", log_dir)

        # Tensorboard callback function
        tensorboard_cb = TensorBoard(log_dir=log_dir, histogram_freq=1)
        callbacks.append(tensorboard_cb)
    
    return callbacks

if __name__ == '__main__':

    # Load Data & Augmentation
    load_data('../run4')
    load_data('../run4_curve')

    X = np.array(data['feature'])
    y = np.array(data['label'])
    X, y = shuffle(X, y)

    # Print out features/labels information after augmentation
    print("data['feature'] shape: {}".format(X.shape))
    print("data['label'] shape: {}".format(y.shape));

    # plot histogram of training/validation set
    # print(np.max(y), np.min(y))
    # plt.hist(y, bins=19)
    # plt.show()

    # Create simple CNN graph
    model = simple_model()

    # Print graph summary
    model.summary()

    # Train network
    model.fit(
        X, y, 
        batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1,
        validation_split=0.2, shuffle=True, 
        callbacks=get_callbacks(early_stop=True, tensorboard_log=True)
    )

    # Save trained network
    model.save('model.h5')