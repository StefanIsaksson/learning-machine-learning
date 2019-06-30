# Identifying hand-written numbers using TensorFlow with Keras API
This repository contains example code to try out TensorFlow with Keras API on the MNIST dataset.

The code is following along an [official tutorial](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/basic_classification.ipynb).

To understand the code I also recommend the following accompanied video: [Get started with TensorFlow's High-Level APIs](https://www.youtube.com/watch?v=tjsHSIG8I08).

## Prerequisites

[Anaconda](https://www.anaconda.com/)

## Install
Open `Anaconda Prompt`.

Create a new conda environment  with Tensorflow. 
```
conda create -n tensorflow_mnist_env tensorflow
```

Activate the new environment.
```
activate tensorflow_mnist_env
```

Install the additional library matplotlib
```
conda install matplotlib
```

## Run
In the `Anaconda Prompt` run `python mnist.py`.

To run example in PyCharm change Project Interpreter to the Conda Environement `tensorflow_mnist_env`. 
Then run `python mnist.py`.
I have tried running the code using the version/flavor [PyCharm for Anaconda](https://www.jetbrains.com/pycharm/promo/anaconda/).