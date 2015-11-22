# (MNIST) Handwritten Digits Recognition

Convolutional neural network to recognize handwritten digits (MNIST dataset).

MNIST Dataset is downloaded from `https://s3.amazonaws.com/torch7/data/mnist.t7.tgz` which contains images (**32x32**)
with classes from 1 to 10 divided into **60'000 training images** and **10'000 test images**.
To use this script you have to download [Torch](http://torch.ch/docs/getting-started.html#_) first.

Below you can see example usage of this convolutional neural network.

## Kaggle

You can also use this model to predict output for Kaggle [Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) challenge.
Unfortunately Kaggle MNIST Dataset have different images geometry (**28x28**) and different file format what makes code a little bit uglier :C
Kaggle dataset consist of **42'000 training images** with classes from 0 to 9 and **28'000 images to predict**.
In `cnn_model.lua` you can find exact model which is used in learning.

To use Kaggle model type:

	$ th train.lua -kaggle 1 -train_size 42000 -test_size 28000

**Tip:** after each iteration script saves new predictions into `data/submission.csv` file.

Max score I have achieved on Kaggle (with about 10 minutes of learning): **0.98686**

## Parameters

You can set parameters which will help you to better control the network.

| Parameter | Optional usage | Description | Default |
| :-------: | :------------: | :--------: | :-----: |
| -gpuid | --enable_gpu | Enables CUDA (use only if you have NVIDIA GPU) | -1 |
| -kaggle | --enable_kaggle | Switches to Kaggle challenge prediction | nil |
| -train_size | --traing_data_size | Size of training sets | 60000 |
| -test_size | --test_data_size | Size of testing sets | 10000 |
| -rate | --learning_rate | Learning rate | 0.05 |
| -batch | --batch_size | Numbers of sets in batch | 10 |
| -t | --threads | Number of threads used during usage | 2 |


## Examples

### Normal:

	$ th train.lua --learning_rate 0.01 --batch_size 100
	$ th train.lua --learning_rate 0.02 --threads 8 --traing_data_size 2000 --test_data_size 1000

### Kaggle:

	$ th train.lua -kaggle 1 -train_size 42000 -test_size 28000
	$ th train.lua -kaggle 1 -train_size 42000 -test_size 28000 --learning_rate 0.005 --batch_size 50
