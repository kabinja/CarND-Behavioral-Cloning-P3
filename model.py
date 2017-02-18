import os
import csv
import random

import numpy as np
import cv2
import sklearn
from sklearn.model_selection import train_test_split

from keras.models import Sequential, Model
from keras.layers import Cropping2D, Lambda

# preprocessing helper

def process_image(img):
    crop_img = img[60:140,0:320]
    yuv_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2YUV)
    cv2.normalize(yuv_img, yuv_img, 0, 255, cv2.NORM_MINMAX)

    return yuv_img

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

        images = []
        angles = []

        for batch_sample in batch_samples:
            path = batch_sample[0]
            center_image = process_image(cv2.imread(path))
            center_angle = float(batch_sample[3])

            images.append(center_image)
            angles.append(center_angle)

        X_train = np.array(images)
        y_train = np.array(angles)

        yield sklearn.utils.shuffle(X_train, y_train)

# load images and steering data
samples = []
with open('./examples/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 80, 320

# build model
model = Sequential()

model.add(Lambda(lambda x: x/255 - 0.5, input_shape=[row, col, ch], \
                 output_shape=[row, col, ch]))

#TODO: Build the model using the architecture suggested by the paper

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=\
                    len(train_samples), validation_data=validation_generator, \
                    nb_val_samples=len(validation_samples), nb_epoch=3)

