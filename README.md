# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

This is a project for Udacity Self-Drving Car NanoDegree. The aim for this project is to control a car in a simulator using neural network. The implementation uses a convolutional neural network (CNN).

## Dependencies

- Simulator provided by Udacity contained in [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

## How to run the project

``` bash
python drive.py model.h5
```
Open the simulator and select **"Autonomous Mode"**

## Architecture

In **training** mode, user can drive the car around the track manually and record the camera images captured by center, left and right camera. 

![Camera Setup](/demo_imgs/CameraSetup.png)

In **Autonomous** mode, the simulator provides `drive.py` a continuous feed of the image from the center camera in size of `160x320`. `drive.py` will run the trained CNN defined by `model.h5` to predict the steering command, which will be sent to the simulator to control the vehicle.

## CNN Architecture

Overview of the CNN layers provided in `TensorBoard` is as follows:

![CNN Graph](./demo_imgs/tensorboard_graph.png)

Details of each later is defined in `simple_mode.py`. The input shape of the CNN is `16x32` and the first layer is to normalize the input to `[-1,1]`. 2 layers of `Dropout` is used to prevent overfitting. 

``` python
model = Sequential(
    [Lambda(lambda x: x/127.5 - 1, input_shape=(img_rows,img_cols,1)),
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
```

Summary of each layer is as follow:

``` shell
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lambda (Lambda)              (None, 16, 32, 1)         0
_________________________________________________________________
conv2d (Conv2D)              (None, 14, 30, 16)        160
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 7, 15, 16)         0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 5, 13, 32)         4640
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 2, 6, 32)          0
_________________________________________________________________
dropout (Dropout)            (None, 2, 6, 32)          0
_________________________________________________________________
flatten (Flatten)            (None, 384)               0
_________________________________________________________________
dense (Dense)                (None, 100)               38500
_________________________________________________________________
dropout_1 (Dropout)          (None, 100)               0
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 101
=================================================================
Total params: 43,401
Trainable params: 43,401
Non-trainable params: 0
_________________________________________________________________
```

## Training

Total of images are used for the training. With image augmentation, the final training/validation set is

``` shell
Train on 96422 samples, validate on 24106 samples
```

I chose the popular `Adam` optimizer with an initial learning rate of `0.001` and `decay = 0.1`. Training is set up to train `15` epochs with an `EarlyStop` callback of patience `3`, and the training ended at epoch `14`. The validation loss over epochs is shown below:

![validation loss](./demo_imgs/val_loss.png)
