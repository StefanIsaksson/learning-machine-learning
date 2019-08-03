import numpy as np
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

train_datagen=ImageDataGenerator(rescale=1./255., preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory("male_vs_female/dataset/train",
                                                    target_size=(150,150),
                                                    color_mode="rgb",
                                                    batch_size=16,
                                                    class_mode="categorical")

valid_datagen=ImageDataGenerator(rescale=1./255., preprocessing_function=preprocess_input)

valid_generator = valid_datagen.flow_from_directory("male_vs_female/dataset/valid",
                                                    target_size=(150,150),
                                                    color_mode="rgb",
                                                    batch_size=16,
                                                    class_mode="categorical")

step_size_train=train_generator.n//train_generator.batch_size
step_size_valid=valid_generator.n//valid_generator.batch_size

base_model = VGG16(weights="imagenet", include_top=False, input_shape=(150, 150, 3))

print(base_model.summary())

model_t1 = Sequential()
model_t1.add(base_model)
model_t1.add(Flatten())
model_t1.add(Dense(64, activation="relu"))
model_t1.add(Dense(2, activation="softmax"))

print(model_t1.summary())

base_model.trainable=False
model_t1.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model_t1.fit_generator(train_generator,
                    steps_per_epoch=step_size_train,
                    epochs=10,
                    validation_data=valid_generator,
                    validation_steps=step_size_valid)

model_t1.save('male_vs_female_model_USING_transfer_learning.h5')