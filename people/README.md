# Image classification of female and male portraits using Keras (Tensorflow) and Transfer Learning

## Prerequisites

[Anaconda](https://www.anaconda.com/)

## Install
Open `Anaconda Prompt`.

Create a new conda environment  with Tensorflow. 
```
conda create -n keras_tf_env pillow tensorflow keras
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


## Measured Model Accuracy

|Model                                           | Accuracy |
|------------------------------------------------|----------|
|Transfer learning model stats (2 epochs only ..)|0.8522    |
|No Transfer learning model stats (10 epochs)    |0.8130    |