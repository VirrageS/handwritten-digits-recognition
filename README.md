# (MNIST) Handwritten Digits Recognition

Convolution neural network to recognize handwritten digits (MNIST dataset).

# Usage

To use this network, you need download [Torch](http://torch.ch/docs/getting-started.html#_) first.

After you do that the next step is just use this:

	th train.lua

## Parameters

You can set parameters which will help you to better control the network.

Parameters such as:

| Parameter | Optional usage | Description | Default |
| :-------: | :------------: | :--------: | :-----: |
| -train_size | --traing_data_size | Size of training sets | 60000 |
| -test_size | --test_data_size | Size of testing sets | 10000 |
| -rate | --learning_rate | Learning rate | 0.05 |
| -batch | --batch_size | Numbers of sets in batch | 10 |
| -t | --threads | Number of threads used during usage | 2 |


### Examples

	$ th train.lua --learning_rate 0.01 --batch_size 100


	$ th train.lua --learning_rate 0.02 --threads 8 --traing_data_size 2000 --test_data_size 1000
