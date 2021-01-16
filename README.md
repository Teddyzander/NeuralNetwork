# NeuralNetwork
Short neural network project

Creates a dynamic Neural Network class to use a feed-forward algorithm to classify data.

Errors are propogated back, with weights and biases being altered using stochastic gradient descent and some learning rate.

A network is created using the Network class, passing in a network structure to the ClassifyTestData function, eg

std::vector<unsigned> nneurons = {2, 3, 3, 1 };
ClassifyTestData(nneurons, "Network", 1);

The above example creates a network with 2 input neurons, 2 hidden layers each with 3 neurons, and one output neuron. 
It will use data set 1 for this network, and save the training data set as "test_pointsNetwork.txt"
and network result as "test_contourNetwork.txt"

As an example, using the spiral data set and a network of { 2, 15, 15, 15, 1 }, the following plot is the resulting network

![Spiral](https://github.com/Teddyzander/NeuralNetwork/blob/master/Spiral21515151(0.06).png)

This image shows that the neural network has divided the R^2 plane into two areas - areas of success (training data set = 1, white areas),
and areas of failure (training data set = -1, black areas). It mostly matches the training data set.

The same was true for the checkerboard data set

![checker](https://github.com/Teddyzander/NeuralNetwork/blob/master/Checker21515151(0.02).png)
