import argparse
from sys import argv
from keras.models import load_model
from keras.preprocessing import image
import numpy as np


def predict_gender(input_file):
    model = load_model('male_vs_female_model_WITHOUT_using_transfer_learning.h5')

    img_width, img_height = 150, 150
    img = image.load_img(input_file, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict_classes(images, batch_size=10)

    prediction_map = {
        1 : 'female',
        0 : 'male'
    }

    gender_prediction = prediction_map[classes[0]]
    return gender_prediction


def main(input_file=None):
    prediction = predict_gender(input_file)
    print(f'"{input_file}" predicted to be {prediction}')
    return prediction


def parse_args(args):
    parser = argparse.ArgumentParser(description='Classifies human portrait as either male or female)')
    parser.add_argument('input_file', help='input file, e.g. ./example_input/female.png')
    return parser.parse_args(args)


if __name__ == '__main__':
    main(**parse_args(argv[1:]).__dict__)
