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

