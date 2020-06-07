import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow_core.python.keras import Model, Sequential
from tensorflow_core.python.keras.api._v2.keras.layers import Lambda, MaxPooling2D, Dropout, Flatten, Dense, Conv2D, Input
from tensorflow_core.python.keras.losses import MeanSquaredError
from tensorflow_core.python.keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.utils import shuffle

mydata_path = './run2'
img_rows = 80
img_cols = 160

# load logs
logs = []
with open(mydata_path + '/driving_log.csv','rt') as f:
    reader = csv.reader(f)
    for line in reader:
        # print(line)
        logs.append(line)
    logs.pop(0)

# samples
samples = []
labels = []
steering_offset = 0.25

def preprocess_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = cv2.resize(img, (img_cols, img_rows)) # resize expected (width, height)
    _,s,_ = cv2.split(img)
    s = np.reshape(s, (img_rows, img_cols, 1))
    return s

def load_data():
    #load center -> left -> right images
    for i in range(3):
        for j in range(len(logs)):
        # for j in range(1000):
            if float(logs[j][3]) <= 0.001:
                continue
            img_filename = logs[j][i].split('IMG')[-1].strip()
            # print(img_filename)
            img_path = mydata_path + '/IMG' + img_filename
            pp_img = preprocess_img(img_path)
            samples.append(pp_img)
            samples.append(np.fliplr(pp_img))
            if i == 0:
                # center image
                labels.append(float(logs[j][3]))
                labels.append(-float(logs[j][3]))
            elif i == 1:
                # left image
                labels.append(float(logs[j][3]) + steering_offset)
                labels.append(-float(logs[j][3]) - steering_offset)
            else:
                # right image
                labels.append(float(logs[j][3]) - steering_offset) 
                labels.append(-float(logs[j][3]) + steering_offset) 

# img = cv2.imread("C:\\Users\\Chishing\\Desktop\\beta_simulator_windows\\mydata\\IMG\\center_2020_06_07_00_07_21_208.jpg")
# print(img.shape)
# # cv2.imshow('image',img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# img = cv2.resize(img, (img_rows, img_cols))
# # cv2.imshow('image',img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# _,s,_ = cv2.split(img)
# # print(s)
# # a = np.reshape(s, (16,32,1))
# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# cv2.imshow('image',s)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print(a)

load_data()
X = np.array(samples)
print("samples shape: {}".format(X.shape))
y = np.array(labels)
print("labels shape: {}".format(y.shape));
X, y = shuffle(X, y)

plt.hist(y, bins=19)
plt.show()

model = Sequential([
			Lambda(lambda x: x/255 - 0.5, input_shape=(img_rows,img_cols,1)),
			Conv2D(filters=16, kernel_size=5, strides=1, padding='valid', activation='relu'),
			MaxPooling2D((4,4),(4,4),'valid'),
			Dropout(0.25),
			Flatten(),
			Dense(1)
		])

model.summary()
model.compile(optimizer='adam', loss='mean_squared_error')

early_stop_cb = EarlyStopping(monitor='loss', patience=3)
model.fit(X, y, batch_size=32, epochs=50, verbose=1,
            validation_split=0.2, shuffle=True, callbacks=[early_stop_cb])
model.save('model.h5')