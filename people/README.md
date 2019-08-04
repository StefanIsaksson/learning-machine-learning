# Image classification of female and male portraits using Keras (Tensorflow) and Transfer Learning

## Prerequisites

[Anaconda](https://www.anaconda.com/)

## Install
Open `Anaconda Prompt`.

Create a new conda environment  with Tensorflow. 
```
conda create -n keras_tf_env pydot pillow tensorflow keras
```

### To also run download_and_classify_random_face_pics.py

```
conda install opencv
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

|Model                                           |Train Accuracy| Valildation Accuracy |
|------------------------------------------------|--------------|----------------------|
|Transfer learning model stats                   |0.9987        |0.8814                |
|No Transfer learning model stats (10 epochs)    |              |0.8130                |


20 killar - av dem är 4 fel. 16/20 = 80% rätt
20 tjejer - av dem är 4 fel. 16/20 = 80% rätt