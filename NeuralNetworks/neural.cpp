
#include "mvector.h"
#include "mmatrix.h"

#include <cmath>
#include <random>
#include <iostream>
#include <fstream>
#include <cassert>
#include <string> 


////////////////////////////////////////////////////////////////////////////////
// Set up random number generation

// Set up a "random device" that generates a new random number each time the program is run
std::random_device rand_dev;

// Set up a pseudo-random number generater "rnd", seeded with a random number
std::mt19937 rnd(rand_dev());

// Alternative: set up the generator with an arbitrary constant integer. This can be useful for
// debugging because the program produces the same sequence of random numbers each time it runs.
// To get this behaviour, uncomment the line below and comment the declaration of "rnd" above.
//std::mt19937 rnd(12345);


////////////////////////////////////////////////////////////////////////////////
// Some operator overloads to allow arithmetic with MMatrix and MVector.
// These may be useful in helping write the equations for the neural network in
// vector form without having to loop over components manually. 
//
// You may not need to use all of these; conversely, you may wish to add some
// more overloads.

// MMatrix * MVector
MVector operator*(const MMatrix &m, const MVector &v)
{
	assert(m.Cols() == v.size());

	MVector r(m.Rows());

	for (int i=0; i<m.Rows(); i++)
	{
		for (int j=0; j<m.Cols(); j++)
		{
			r[i]+=m(i,j)*v[j];
		}
	}
	return r;
}

// transpose(MMatrix) * MVector
MVector TransposeTimes(const MMatrix &m, const MVector &v)
{
	assert(m.Rows() == v.size());

	MVector r(m.Cols());

	for (int i=0; i<m.Cols(); i++)
	{
		for (int j=0; j<m.Rows(); j++)
		{
			r[i]+=m(j,i)*v[j];
		}
	}
	return r;
}

// MVector + MVector
MVector operator+(const MVector &lhs, const MVector &rhs)
{
	assert(lhs.size() == rhs.size());

	MVector r(lhs);
	for (int i=0; i<lhs.size(); i++)
		r[i] += rhs[i];

	return r;
}

// MVector - MVector
MVector operator-(const MVector &lhs, const MVector &rhs)
{
	assert(lhs.size() == rhs.size());

	MVector r(lhs);
	for (int i=0; i<lhs.size(); i++)
		r[i] -= rhs[i];

	return r;
}

// MMatrix = MVector <outer product> MVector
// M = a <outer product> b
MMatrix OuterProduct(const MVector &a, const MVector &b)
{
	MMatrix m(a.size(), b.size());
	for (int i=0; i<a.size(); i++)
	{
		for (int j=0; j<b.size(); j++)
		{
			m(i,j) = a[i]*b[j];
		}
	}
	return m;
}

// Hadamard product
MVector operator*(const MVector &a, const MVector &b)
{
	assert(a.size() == b.size());
	
	MVector r(a.size());
	for (int i=0; i<a.size(); i++)
		r[i]=a[i]*b[i];
	return r;
}

// double * MMatrix
MMatrix operator*(double d, const MMatrix &m)
{
	MMatrix r(m);
	for (int i=0; i<m.Rows(); i++)
		for (int j=0; j<m.Cols(); j++)
			r(i,j)*=d;

	return r;
}

// double * MVector
MVector operator*(double d, const MVector &v)
{
	MVector r(v);
	for (int i=0; i<v.size(); i++)
		r[i]*=d;

	return r;
}

// MVector -= MVector
MVector operator-=(MVector &v1, const MVector &v)
{
	assert(v1.size()==v.size());
	
	for (int i=0; i<v1.size(); i++)
		v1[i]-=v[i];
	
	return v1;
}

// MMatrix -= MMatrix
MMatrix operator-=(MMatrix &m1, const MMatrix &m2)
{
	assert (m1.Rows() == m2.Rows() && m1.Cols() == m2.Cols());

	for (int i=0; i<m1.Rows(); i++)
		for (int j=0; j<m1.Cols(); j++)
			m1(i,j)-=m2(i,j);

	return m1;
}

// Output function for MVector
inline std::ostream &operator<<(std::ostream &os, const MVector &rhs)
{
	std::size_t n = rhs.size();
	os << "(";
	for (std::size_t i=0; i<n; i++)
	{
		os << rhs[i];
		if (i!=(n-1)) os << ", ";
	}
	os << ")";
	return os;
}

// Output function for MMatrix
inline std::ostream &operator<<(std::ostream &os, const MMatrix &a)
{
	int c = a.Cols(), r = a.Rows();
	for (int i=0; i<r; i++)
	{
		os<<"(";
		for (int j=0; j<c; j++)
		{
			os.width(10);
			os << a(i,j);
			os << ((j==c-1)?')':',');
		}
		os << "\n";
	}
	return os;
}

//struct for returning stuff needed for plots
struct NetworkData
{
	bool success;
	double eta;
	int iterations;
	double cost;
};


////////////////////////////////////////////////////////////////////////////////
// Functions that provide sets of training data

// Generate 16 points of training data in the pattern illustrated in the project description
void GetTestData(std::vector<MVector> &x, std::vector<MVector> &y)
{
	x = {{0.125,.175}, {0.375,0.3125}, {0.05,0.675}, {0.3,0.025}, {0.15,0.3}, {0.25,0.5}, {0.2,0.95}, {0.15, 0.85},
		 {0.75, 0.5}, {0.95, 0.075}, {0.4875, 0.2}, {0.725,0.25}, {0.9,0.875}, {0.5,0.8}, {0.25,0.75}, {0.5,0.5}};
	
	y = {{1},{1},{1},{1},{1},{1},{1},{1},
		 {-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1}};
}

// Generate 1000 points of test data in a checkerboard pattern
void GetCheckerboardData(std::vector<MVector> &x, std::vector<MVector> &y)
{
	std::mt19937 lr;
	x = std::vector<MVector>(1000, MVector(2));
	y = std::vector<MVector>(1000, MVector(1));

	for (int i=0; i<1000; i++)
	{
		x[i]={lr()/static_cast<double>(lr.max()),lr()/static_cast<double>(lr.max())};
		double r = sin(x[i][0]*12.5)*sin(x[i][1]*12.5);
		y[i][0] = (r>0)?1:-1;
	}
}


// Generate 1000 points of test data in a spiral pattern
void GetSpiralData(std::vector<MVector> &x, std::vector<MVector> &y)
{
	std::mt19937 lr;
	x = std::vector<MVector>(1000, MVector(2));
	y = std::vector<MVector>(1000, MVector(1));

	double twopi = 8.0*atan(1.0);
	for (int i=0; i<1000; i++)
	{
		x[i]={lr()/static_cast<double>(lr.max()),lr()/static_cast<double>(lr.max())};
		double xv=x[i][0]-0.5, yv=x[i][1]-0.5;
		double ang = atan2(yv,xv)+twopi;
		double rad = sqrt(xv*xv+yv*yv);

		double r=fmod(ang+rad*20, twopi);
		y[i][0] = (r<0.5*twopi)?1:-1;
	}
}

// Save the the training data in x and y to a new file, with the filename given by "filename"
// Returns true if the file was saved succesfully
bool ExportTrainingData(const std::vector<MVector> &x, const std::vector<MVector> &y,
						std::string filename)
{
	// Check that the training vectors are the same size
	assert(x.size()==y.size());

	// Open a file with the specified name.
	std::ofstream f(filename); 

	// Return false, indicating failure, if file did not open
	if (!f)
	{
		return false;
	}

	// Loop over each training datum
	for (unsigned i=0; i<x.size(); i++)
	{
		// Check that the output for this point is a scalar
		assert(y[i].size() == 1);
		
		// Output components of x[i]
		for (int j=0; j<x[i].size(); j++)
		{
			f << x[i][j] << " ";
		}

		// Output only component of y[i]
		f << y[i][0] << " " << std::endl;
	}
	f.close();

	if (f) return true;
	else return false;
}




////////////////////////////////////////////////////////////////////////////////
// Neural network class

class Network
{
public:

	// Constructor: sets up vectors of MVectors and MMatrices for
	// weights, biases, weighted inputs, activations and errors
	// The parameter nneurons_ is a vector defining the number of neurons at each layer.
	// For example:
	//   Network({2,1}) has two input neurons, no hidden layers, one output neuron
	//
	//   Network({2,3,3,1}) has two input neurons, two hidden layers of three neurons
	//                      each, and one output neuron
	Network(std::vector<unsigned> nneurons_)
	{
		nneurons = nneurons_;
		nLayers = nneurons.size();
		weights = std::vector<MMatrix>(nLayers); 
		biases = std::vector<MVector>(nLayers); 
		errors = std::vector<MVector>(nLayers);
		activations = std::vector<MVector>(nLayers); 
		inputs = std::vector<MVector>(nLayers);
		// Create activations vector for input layer 0
		activations[0] = MVector(nneurons[0]);

		// Other vectors initialised for second and subsequent layers
		for (unsigned i=1; i<nLayers; i++)
		{
			weights[i] = MMatrix(nneurons[i], nneurons[i-1]);
			biases[i] = MVector(nneurons[i]);
			inputs[i] = MVector(nneurons[i]);
			errors[i] = MVector(nneurons[i]);
			activations[i] = MVector(nneurons[i]);
		}

		// The correspondence between these member variables and
		// the LaTeX notation used in the project description is:
		//
		// C++                      LaTeX
		// -------------------------------------
		// inputs[l-1][j-1]      =  z_j^{[l]}
		// activations[l-1][j-1] =  a_j^{[l]}
		// weights[l-1](j-1,k-1) =  W_{jk}^{[l]}
		// biases[l-1][j-1]      =  b_j^{[l]}
		// errors[l-1][j-1]      =  \delta_j^{[l]}
		// nneurons[l-1]         =  n_l
		// nLayers               =  L
		//
		// Note that, since C++ vector indices run from 0 to N-1, all the indices in C++
		// code are one less than the indices used in the mathematics (which run from 1 to N)
	}

	// Return the number of input neurons
	unsigned NInputNeurons() const
	{
		return nneurons[0];
	}

	// Return the number of output neurons
	unsigned NOutputNeurons() const
	{
		return nneurons[nLayers-1];
	}
	
	// Evaluate the network for an input x and return the activations of the output layer
	MVector Evaluate(const MVector &x)
	{
		// Call FeedForward(x) to evaluate the network for an input vector x
		FeedForward(x);

		// Return the activations of the output layer
		return activations[nLayers-1];
	}

	
	// Implement the training algorithm outlined in section 1.3.3
	// This should be implemented by calling the appropriate private member functions, below
	NetworkData Train(const std::vector<MVector> x, const std::vector<MVector> y,
			   double initsd, double learningRate, double costThreshold, int maxIterations)
	{
		// Check that there are the same number of training data inputs as outputs
		assert(x.size() == y.size());

		//initialise the weights and biases with the standard deviation "initsd"
		InitialiseWeightsAndBiases(initsd);
		
		for (int iter=1; iter<=maxIterations; iter++)
		{
			// Step 3: Choose a random training data point i in {0, 1, 2, ..., N}
			int i = rnd()%x.size();

			// Step 4: run the feed-forward algorithm
			FeedForward(x[i]);

			//Step 5: run the back-propagation algorithm
			BackPropagateError(y[i]);
			
			// Step 6: update the weights and biases using stochastic gradient
			//                with learning rate "learningRate"
			UpdateWeightsAndBiases(learningRate);

			// Every so often, perform step 7 and show an update on how the cost function has decreased
			// Here, "every so often" means once every 1000 iterations, and also at the last iteration
			if ((!(iter%1000)) || iter==maxIterations)
			{
				// Step 7(a): calculate the total cost
				double total_cost = TotalCost(x, y);


				// display the iteration number and total cost to the screen
				std::cout << "Iteration: " << iter << "\tTotal Cost:" << total_cost << std::endl;
				
				// Step 7(b): return from this method with a value of true,
	 			//                   indicating success, if this cost is less than "costThreshold".

				if (total_cost < costThreshold)
				{
					return { true, learningRate, iter, total_cost };
				}
			}
			
		} // Step 8: go back to step 3, until we have taken "maxIterations" steps

		// Step 9: return "false", indicating that the training did not succeed.
		return { false, learningRate, maxIterations, TotalCost(x, y) };
	}

	
	// For a neural network with two inputs x=(x1, x2) and one output y,
	// loop over (x1, x2) for a grid of points in [0, 1]x[0, 1]
	// and save the value of the network output y evaluated at these points
	// to a file. Returns true if the file was saved successfully.
	bool ExportOutput(std::string filename)
	{
		// Check that the network has the right number of inputs and outputs
		assert(NInputNeurons()==2 && NOutputNeurons()==1);
	
		// Open a file with the specified name.
		std::ofstream f(filename); 
	
		// Return false, indicating failure, if file did not open
		if (!f)
		{
			return false;
		}

		// generate a matrix of 250x250 output data points
		for (int i=0; i<=250; i++)
		{
			for (int j=0; j<=250; j++)
			{
				MVector out = Evaluate({i/250.0, j/250.0});
				f << out[0] << " ";
			}
			f << std::endl;
		}
		f.close();
	
		if (f) return true;
		else return false;
	}


	static bool Test(int test);
	
private:
	// Return the activation function sigma
	/*
	Activation function that returns value between [-1, 1]. The more extreme the input,
	the closer the returned value will be to 1, with extreme negatives ~ -1 and extreme
	positives ~ 1
	*/
	double Sigma(double z)
	{
		return tanh(z);
	}

	// Return the derivative of the activation function
	/*
	Derivative of tanh(x) = sech^2(x) = 1 / cosh^2(x)
	*/
	double SigmaPrime(double z)
	{
		double temp = cosh(z);
		return (1.0 / (temp * temp));
	}
	
	// Loop over all weights and biases in the network and set each
	// term to a random number normally distributed with mean 0 and
	// standard deviation "initsd"
	void InitialiseWeightsAndBiases(double initsd)
	{
		// Make sure the standard deviation supplied is non-negative
		assert(initsd>=0);

		// Set up a normal distribution with mean zero, standard deviation "initsd"
		// Calling "dist(rnd)" returns a random number drawn from this distribution 
		std::normal_distribution<> dist(0, initsd);

		// go through each layer (except the first)
		for (int i = 1; i < biases.size(); i++)
		{
			// at each layer, the biases are a vector of values equal to the number of neurons
			// at each layer, the weights are a matrix of values equal to the 
			// number of neurons x number of neurons in previous layer, so we can go through
			// these data structures simulatenously 

			// set all weights in this layer
			weights[i] = dist(rnd);

			// go through all biases in this layer and set them
			for (int j = 0; j < biases[i].size(); j++)
			{
				biases[i][j] = dist(rnd);
			}
		}

	}

	// Evaluate the feed-forward algorithm, setting weighted inputs and activations
	// at each layer, given an input vector x
	void FeedForward(const MVector &x)
	{
		// Check that the input vector has the same number of elements as the input layer
		assert(x.size() == nneurons[0]);
		
		// TODO: Implement the feed-forward algorithm, equations (1.7), (1.8)

		// use the set values in the activation paramater
		// first layer is equal to the input
		activations[0] = x;

		// go through each layer, excluding the first layer
		for (int l = 1; l < nLayers; l++)
		{
			int prev_l = l - 1;

			// get our inputs (z) for the activation function (sigma) 
			inputs[l] = weights[l] * activations[prev_l] + biases[l];

			// for each neuron in the current layer put the inputs through sigma
			for (int n = 0; n < inputs[l].size(); n++)
			{
				activations[l][n] = Sigma(inputs[l][n]);
			}
		}
	}

	// Evaluate the back-propagation algorithm, setting errors for each layer 
	void BackPropagateError(const MVector &y)
	{
		// Check that the output vector y has the same number of elements as the output layer
		assert(y.size() == nneurons[nLayers - 1]);

		// TODO: Implement the back-propagation algorithm, equations (1.22) and (1.24)

		// The error is  vector of vectors, where each layer has an error for each nueron.
		// We must evaluate back to front, neurons to find errors for the previous neuron
		
		// evaluate final layer

		MVector diff = activations[nLayers - 1] - y;
		std::vector<MVector> temp = inputs; // temp vector for signma prime
		
		for (int n = 0; n < inputs[nLayers - 1].size(); n++)
		{
			temp[nLayers - 1][n] = SigmaPrime(inputs[nLayers - 1][n]);
		}
		
		errors[nLayers - 1] = temp[nLayers - 1] * diff;

		// now we can use the final layer to propogate the errors back through the network
		// No error associated with the input layer, so end after l == 1.
		for (int l = nLayers - 2; l > 0; l--)
		{

			for (int n = 0; n < inputs[l].size(); n++)
			{
				temp[l][n] = SigmaPrime(inputs[l][n]);
			}
			MVector temp2 = TransposeTimes(weights[l + 1], errors[l + 1]);
			errors[l] = temp[l] * temp2;
		}
	}

	
	// Apply one iteration of the stochastic gradient iteration with learning rate eta.
	void UpdateWeightsAndBiases(double eta)
	{
		// Check that the learning rate is positive
		assert(eta>0);
		
		// TODO: update the weights and biases according to the stochastic gradient
		//       iteration, using equations (1.25) and (1.26) to evaluate
		//       the components of grad C.

		// go through each layer and update weights and biases, excluding the first layer

		for (int l = 1; l < nLayers; l++)
		{
			biases[l] -= eta * errors[l];
			weights[l] -= eta * OuterProduct(errors[l], activations[l - 1]);
		}
	}

	
	// Return the cost function of the network with respect to a single the desired output y
	// Note: call FeedForward(x) first to evaluate the network output for an input x,
	//       then call this method Cost(y) with the corresponding desired output y
	double Cost(const MVector &y)
	{
		// Check that y has the same number of elements as the network has outputs
		assert(y.size() == nneurons[nLayers-1]);
		
		// TODO: Return the cost associated with this output
		double c = 0;
		
		// find L_2 of y - a^{L} x where y is the output, x is the input, 
		//a^{L} are the activations in the final layer of neurons

		for (int i = 0; i < y.size(); i++)
		{
			c += std::pow(y[i] - activations[nLayers - 1][i], 2);
		}

		return 0.5 * c;
	}

	// Return the total cost C for a set of training data x and desired outputs y
	double TotalCost(const std::vector<MVector> x, const std::vector<MVector> y)
	{
		// Check that there are the same number of inputs as outputs
		assert(x.size() == y.size());

		// TODO: Implement the cost function, equation (1.9), using
		//       the FeedForward(x) and Cost(y) methods

		double total_cost = 0;

		for (int i = 0; i < x.size(); i++)
		{
			FeedForward(x[i]);
			total_cost += Cost(y[i]);
		}
		
		return (1.0 / x.size()) * total_cost;
	}

	// Private member data
	
	std::vector<unsigned> nneurons;
	std::vector<MMatrix> weights;
	std::vector<MVector> biases, errors, activations, inputs;
	unsigned nLayers;

};



bool Network::Test(int test = 0)
{
	// This function is a static member function of the Network class:
	// it acts like a normal stand-alone function, but has access to private
	// members of the Network class. This is useful for testing, since we can
	// examine and change internal class data.
	//
	// This function should return true if all tests pass, or false otherwise

	double tol = 1e-10;

	// A example test of FeedForward
	if (test == 1 || test == 0)
	{
		// Make a simple network with two weights and one bias
		Network n({2, 1});

		// Set the values of these by hand
		n.biases[1][0] = 0.5;
		n.weights[1](0,0) = -0.3;
		n.weights[1](0,1) = 0.2;

		// Call function to be tested with x = (0.3, 0.4)
		n.FeedForward({0.3, 0.4});

		// Display the output value calculated
		std::cout << n.activations[1][0] << std::endl;

		// Correct value is = tanh(0.5 + (-0.3*0.3 + 0.2*0.4))
		//                    = 0.454216432682259...
		// Fail if error in answer is greater than 10^-10:
		if (std::abs(n.activations[1][0] - 0.454216432682259) > tol)
		{
			std::cout << "FAILED: FeedForward" << std::endl;
			return false;
		}
	}

	
	// TODO: for each part of the Network class that you implement,
	//       write some more tests here to run that code and verify that
	//       its output is as you expect.
	//       I recommend putting each test in an empty scope { ... }, as 
    //       in the example given above.

	// test activation function
	if (test == 2 || test == 0)
	{
		Network n({ 2, 1 });

		double result = n.Sigma(1000);
		double expect = 1;

		if (std::abs(result - expect) > tol)
		{
			std::cout << "FAILED: activation function, extreme positive" << std::endl;
			return false;
		}

		result = n.Sigma(-1000);
		expect = -1;

		if (std::abs(result - expect) > tol)
		{
			std::cout << "FAILED: activation function, extreme negative" << std::endl;
			return false;
		}

		result = n.Sigma(0);
		expect = 0;

		if (std::abs(result - expect) > tol)
		{
			std::cout << "FAILED: activation function, zero input" << std::endl;
			return false;
		}

		result = n.Sigma(1);
		expect = 0.7615941559557;

		if (std::abs(result - expect) > tol)
		{
			std::cout << "FAILED: activation function, realistic positive input" << std::endl;
			return false;
		}

		result = n.Sigma(-0.3);
		expect = -0.29131261245159;

		if (std::abs(result - expect) > tol)
		{
			std::cout << "FAILED: activation function, realistic negative input" << std::endl;
			return false;
		}

	}

	// test derivative of activation function
	if (test == 3 || test == 0)
	{
		Network n({ 2, 1 });

		double result = n.SigmaPrime(0);
		double expect = 1;

		if (std::abs(result - expect) > tol)
		{
			std::cout << "FAILED: activation function derivative, 0 input" << std::endl;
			return false;
		}

		result = n.SigmaPrime(1000);
		expect = 0;

		if (std::abs(result - expect) > tol)
		{
			std::cout << "FAILED: activation function derivative, extreme positive" << std::endl;
			return false;
		}

		result = n.SigmaPrime(-1000);
		expect = 0;

		if (std::abs(result - expect) > tol)
		{
			std::cout << "FAILED: activation function derivative, extreme negative" << std::endl;
			return false;
		}

		result = n.SigmaPrime(-0.8);
		expect = 0.55905516773;

		if (std::abs(result - expect) > tol)
		{
			std::cout << "FAILED: activation function derivative, extreme negative" << std::endl;
			return false;
		}

		result = n.SigmaPrime(0.4);
		expect = 0.85563878608;

		if (std::abs(result - expect) > tol)
		{
			std::cout << "FAILED: activation function derivative, extreme negative" << std::endl;
			return false;
		}
	}

	// test initialisation if weights and biases
	if (test == 4 || test == 0)
	{
		// create a simple network
		Network n({ 2, 3, 3, 1 });
		double expect = 0;

		// check that weights and biases are all 0 before initialisation. Display them
		for (int i = 1; i < n.biases.size(); i++)
		{
			for (int j = 0; j < n.biases[i].size(); j++)
			{
				if (n.biases[i][j] != expect)
				{
					std::cout << "FAILED: weights and biases pre-initilasation, biases" << std::endl;
					return false;
				}
				for (int k = 0; k < n.weights[i].Cols(); k++)
				{
					if (n.weights[i](j, k) != expect)
					{
						std::cout << "FAILED: weights and biases pre-initilasation, weights" << std::endl;
						return false;
					}
				}
			}
		}

		// initalise weights and biases to a SD of 10 and check none are 0
		n.InitialiseWeightsAndBiases(10);

		for (int i = 1; i < n.biases.size(); i++)
		{
			std::cout << "For layer " << i + 1 << std::endl;
			for (int j = 0; j < n.biases[i].size(); j++)
			{
				std::cout << "For neuron " << j + 1 << std::endl;
				std::cout << "Biase: " << n.biases[i][j] << std::endl;
				if (n.biases[i][j] == expect)
				{
					std::cout << "FAILED: weights and biases initilasation, biases" << std::endl;
					return false;
				}
				for (int k = 0; k < n.weights[i].Cols(); k++)
				{
					std::cout << "Weights: " << n.weights[i](j, k) << std::endl;
					if (n.weights[i](j, k) == expect)
					{
						std::cout << "FAILED: weights and biases initilasation, weights" << std::endl;
						return false;
					}
				}
			}
		}

		// create a new, larger, more complex network, test size of each step is correct
		unsigned int input_size = 2;
		unsigned int layer_size = 10;
		unsigned int output_size = 1;
		Network n2({ input_size, layer_size, layer_size - 1, layer_size + 1, output_size });
		n2.InitialiseWeightsAndBiases(5);
		int loops = 0;
		int expected_loops = (input_size * layer_size) +
			(layer_size * (layer_size - 1)) +
			((layer_size - 1) * (layer_size + 1)) +
			((layer_size + 1) * output_size);

		for (int i = 1; i < n2.biases.size(); i++)
		{
			for (int j = 0; j < n2.biases[i].size(); j++)
			{
				if (n2.biases[i][j] == expect)
				{
					std::cout << "FAILED: weights and biases large network initilasation, biases" << std::endl;
					return false;
				}
				for (int k = 0; k < n2.weights[i].Cols(); k++)
				{
					if (n2.weights[i](j, k) == expect)
					{
						std::cout << "FAILED: weights and biases large network initilasation, weights" << std::endl;
						return false;
					}
					loops += 1;
				}
			}
		}
		
		// check we have checked every value
		if (loops != expected_loops)
		{
			std::cout << "FAILED: weights and biases large network initilasation, check all values" << std::endl;
		}
		
	}

	// test back propogation
	if (test == 5 || test == 0)
	{
		// Make a simple network with two weights and one bias
		Network n({ 2, 1 });

		// Set the values of these by hand
		n.biases[1][0] = 0.5;
		n.weights[1](0, 0) = -0.3;
		n.weights[1](0, 1) = 0.2;

		// applying this feed forward gives an output of 
		// approx 0.454216432682259
		n.FeedForward({ 0.3, 0.4 });

		// check that giving 0.454216432682259 as the true value
		// gives a VERY small error
		n.BackPropagateError({ 0.454216432682259 });

		if (std::abs(n.errors[1][0]) > tol)
		{
			std::cout << "FAILED: back propogation, final layer" << std::endl;
			return false;
		}

		// check that giving |y| >> 0.454216432682259 as the true value
		// gives a large error

		n.BackPropagateError({ -99999999 });

		if (std::abs(n.errors[1][0]) < 1000)
		{
			std::cout << "FAILED: back propogation, final layer" << std::endl;
			return false;
		}

		// check that giving |y| = 1 as the true value
		// gives an error of approx -0.43318

		n.BackPropagateError({ 1 });

		if (std::abs(n.errors[1][0] - -0.43318) > 0.00001)
		{
			std::cout << "FAILED: back propogation, final layer" << std::endl;
			return false;
		}
	}

	// test updating of weights and biases
	if (test == 6 || test == 0)
	{
		// Make a simple network with two weights and one bias
		Network n({ 2, 1 });

		// Set the values of these by hand
		n.biases[1][0] = 0.5;
		n.weights[1](0, 0) = -0.3;
		n.weights[1](0, 1) = 0.2;

		n.UpdateWeightsAndBiases(0.5);

		// since no data has beem run, errors should be all 0, so biases and weights
		// should be unchanged

		if (std::abs(n.biases[1][0] - 0.5) > tol)
		{
			std::cout << "FAILED: updating weights and biases, 0 error for biase" << std::endl;
			return false;
		}
		if (std::abs(n.weights[1](0, 0) - -0.3) > tol)
		{
			std::cout << "FAILED: updating weights and biases, 0 error for weights" << std::endl;
			return false;
		}
		if (std::abs(n.weights[1](0, 1) - 0.2) > tol)
		{
			std::cout << "FAILED: updating weights and biases, 0 error for weights" << std::endl;
			return false;
		}

		// run the network with the exact answer in the back propogation
		// should expect no change to weights and biases
		// applying this feed forward gives an output of 
		// approx 0.454216432682259

		n.FeedForward({ 0.3, 0.4 });
		n.BackPropagateError({ 0.454216432682259 });

		n.UpdateWeightsAndBiases(0.1);

		if (std::abs(n.biases[1][0] - 0.5) > tol)
		{
			std::cout << "FAILED: updating weights and biases, exact answer for biase" << std::endl;
			return false;
		}
		if (std::abs(n.weights[1](0, 0) - -0.3) > tol)
		{
			std::cout << "FAILED: updating weights and biases, exact answer for weights" << std::endl;
			return false;
		}
		if (std::abs(n.weights[1](0, 1) - 0.2) > tol)
		{
			std::cout << "FAILED: updating weights and biases, exact answer for weights" << std::endl;
			return false;
		}

		// set some values for activations and errors

		n.activations[0][0] = 0.1;
		n.activations[0][1] = 0.1;
		n.errors[1][0] = 0.1;
		n.UpdateWeightsAndBiases(0.1);

		// should epect biases to change by 0.01
		// should expect weights to change by 0.001

		if (std::abs(n.biases[1][0] - 0.49) > tol)
		{
			std::cout << "FAILED: updating weights and biases, small error for biase" << std::endl;
			return false;
		}
		if (std::abs(n.weights[1](0, 0) - -0.301) > tol)
		{
			std::cout << "FAILED: updating weights and biases, small error for weights" << std::endl;
			return false;
		}
		if (std::abs(n.weights[1](0, 1) - 0.199) > tol)
		{
			std::cout << "FAILED: updating weights and biases, small error for weights" << std::endl;
			return false;
		}
	}

	// test Cost
	if (test == 7 || test == 0)
	{
		// Make a simple network with two weights and one bias
		Network n({ 2, 1 });

		// Set the values of these by hand
		n.biases[1][0] = 0.5;
		n.weights[1](0, 0) = -0.3;
		n.weights[1](0, 1) = 0.2;

		n.FeedForward({ 0.3, 0.4 });

		// cost should be 0

		double cost = n.Cost({ 0.454216432682259 });
		if (std::abs(cost - 0) > tol)
		{
			std::cout << "FAILED: Cost, simple network; exact answer" << std::endl;
			return false;
		}

		// Make a more complex network
		Network n2({ 2, 3, 4, 3 });

		n2.FeedForward({ 0.1, 0.2 });
		n2.activations[3][0] = 1;
		n2.activations[3][1] = 0.5;
		n2.activations[3][2] = 0.1;

		cost = n2.Cost({ 1, 0.5, 0.1 });

		// check that if output is equal to activations, cost is 0
		if (std::abs(cost - 0) > tol)
		{
			std::cout << "FAILED: Cost, complex network; exact answer" << std::endl;
			return false;
		}

		// check that if output is wrong for one neuron we get the right cost 
		//a^L[0] is out by 0.5, so we expect cost to equal 0.5 * (1-0.5)^2

		cost = n2.Cost({ 0.5, 0.5, 0.1 });
		if (std::abs(cost - 0.125) > tol)
		{
			std::cout << "FAILED: Cost, complex network; one wrong neuron" << std::endl;
			return false;
		}

		// check that the costs add up correctly if >1 are wrong
		// first neuron cost is still 0.125
		// second is 0.5 * (0.5 - (0.8))^2 = 0.045
		// third is 0.5 * (0.1 - (0.9))^2 = 0.32
		// expect cost to be 0.49
		cost = n2.Cost({ 0.5, 0.8, 0.9 });
		if (std::abs(cost - 0.49) > tol)
		{
			std::cout << "FAILED: Cost, complex network; many wrong neurons" << std::endl;
			return false;
		}
	}
	// test total cost
	if (test == 8 || test == 0)
	{
		// Make a simple network with two weights and one bias
		Network n({ 2, 1 });

		// set them to correct values
		n.biases[1][0] = 0.5;
		n.weights[1](0, 0) = -0.3;
		n.weights[1](0, 1) = 0.2;
		n.activations[1][0] = 0.454216432682259;

		// Make a simple training data set which holds correct answers
		// expect total cost to be almost 0

		std::vector<MVector> x, y;
		x = { { 0.3, 0.4 } };
		y = { {0.454216432682259} };

		double total_cost = n.TotalCost(x, y);
		if (std::abs(total_cost) > tol)
		{
			std::cout << "FAILED: TotalCost, right input/output" << std::endl;
			return false;
		}

		//expect total cost to be specifc value

		y = { {0.40421643268225} }; //incorrect by 0.05
		double expect = 0.00125;

		total_cost = n.TotalCost(x, y);

		if (std::abs(total_cost - expect) > tol)
		{
			std::cout << "FAILED: TotalCost, wrong input/output" << std::endl;
			return false;
		}

	}
	return true;
}

////////////////////////////////////////////////////////////////////////////////
// Main function and example use of the Network class

// Create, train and use a neural network to classify the data in
// figures 1.1 and 1.2 of the project description.
//
// You should make your own copies of this function and change the network parameters
// to solve the other problems outlined in the project description.
void ClassifyTestData(std::vector<unsigned> nneurons, std::string filename = "", int dataset = 0)
{
	// Create a network with two input neurons, two hidden layers of three neurons, and one output neuron
	Network n(nneurons);

	// Get some data to train the network
	std::vector<MVector> x, y;

	if (dataset == 1)
	{
		GetCheckerboardData(x, y);
	}
	else if (dataset == 2)
	{
		GetSpiralData(x, y);
	}
	else
	{
		GetTestData(x, y);
	}
	
	// Train network on training inputs x and outputs y
	// Numerical parameters are:
	//  initial weight and bias standard deviation = 0.1
	//  learning rate = 0.1
	//  cost threshold = 1e-4
	//  maximum number of iterations = 10000

	/*
	// try different learning rates and see what results we get
	double eta = 0.001;
	std::ofstream myfile;
	myfile.open("eta_conergence_small.txt");

	while (eta <= 0.35)
	{
		Network n1(nneurons);
		NetworkData trainingSucceeded = n1.Train(x, y, 0.1, eta, 1e-4, 100000);

		// If training failed, report this
		if (!trainingSucceeded.success)
		{
			std::cout << "Failed to converge to desired tolerance." << std::endl;
		}

		myfile << trainingSucceeded.eta << "\t" << trainingSucceeded.cost << "\t" <<
			trainingSucceeded.iterations << std::endl;

		eta += 0.001;
	}
	myfile.close();
	*/

	/*
	// try different standard deviations
	double sd = 5;
	std::ofstream myfile;
	myfile.open("sd_conergence_large.txt");

	while (sd <= 10)
	{
		std::cout << "standard deviation: " << sd << std::endl;
		Network n1(nneurons);
		NetworkData trainingSucceeded = n1.Train(x, y, sd, 0.1, 1e-4, 100000);

		// If training failed, report this
		if (!trainingSucceeded.success)
		{
			std::cout << "Failed to converge to desired tolerance." << std::endl;
		}

		myfile << sd << "\t" << trainingSucceeded.cost << "\t" <<
			trainingSucceeded.iterations << std::endl;

		sd += 0.1;
	}
	myfile.close();
	*/
	NetworkData trainingSucceeded = n.Train(x, y, 0.1, 0.005, 0.01, 3000000);

	// If training failed, report this
	if (!trainingSucceeded.success)
	{
		std::cout << "Failed to converge to desired tolerance." << std::endl;
	}

	// Generate some output files for plotting
	ExportTrainingData(x, y, "test_points" + filename + ".txt");
	n.ExportOutput("test_contour" + filename + ".txt");
}


int main()
{
	// Call the test function	
	bool testsPassed = Network::Test();

	// If tests did not pass, something is wrong; end program now
	if (!testsPassed)
	{
		std::cout << "A test failed." << std::endl;
		return 1;
	}

	std::cout << "Tests passed, procede to example program...\n" << std::endl;

	// Tests passed, so run our example program.
	
	//std::vector<unsigned> nneurons = {2, 3, 3, 1 };
	//ClassifyTestData(nneurons);
	
	// no hidden layers
	std::vector<unsigned> nneurons = {2, 1 };
	ClassifyTestData(nneurons, "Fcheckers21", 1);
	ClassifyTestData(nneurons, "Fspiral21", 2);

	// one hidden layer

	for (unsigned int i = 5; i < 21; i += 5)
	{
		std::vector<unsigned> nneurons1 = { 2, i, 1 };
		ClassifyTestData(nneurons1, "Fcheckers2" + std::to_string(i) + "1", 1);
		ClassifyTestData(nneurons1, "Fspiral2" + std::to_string(i) + "1", 2);
	}

	// two hidden layer

	for (unsigned int i = 5; i < 21; i += 5)
	{
		std::vector<unsigned> nneurons1 = { 2, i, i, 1 };
		ClassifyTestData(nneurons1, "Fcheckers2" + std::to_string(i) + std::to_string(i) + "1", 1);
		ClassifyTestData(nneurons1, "Fspiral2" + std::to_string(i) + std::to_string(i) + "1", 2);
	}

	// three hidden layer
	for (unsigned int i = 5; i < 21; i += 5)
	{
		std::vector<unsigned> nneurons1 = { 2, i, i, i, 1 };
		ClassifyTestData(nneurons1, "Fcheckers2" + std::to_string(i) +
			std::to_string(i) + std::to_string(i) + "1", 1);
		ClassifyTestData(nneurons1, "Fspiral2" + std::to_string(i) +
			std::to_string(i) + std::to_string(i) + "1", 2);
	}
	return 0;
}

