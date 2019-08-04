# Image classification of female and male portraits using Keras (Tensorflow) and Transfer Learning

## Prerequisites

[Anaconda](https://www.anaconda.com/)

## Install
Open `Anaconda Prompt`.

Create a new conda environment  with Tensorflow. 
```
conda create -n keras_tf_env opencv pydot pillow tensorflow keras
```

## Usage:

Creates a model file called: `male_vs_female_model_WITHOUT_using_transfer_learning.h5`
```
python create_model_without_transfer_learning.py
```

Read input file and prints prediction if picture is male or female.
```
python predict_female_vs_male.py "example_input/male.png"
```
