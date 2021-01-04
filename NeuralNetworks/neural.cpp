
#include "mvector.h"
#include "mmatrix.h"

#include <cmath>
#include <random>
#include <iostream>
#include <fstream>
#include <cassert>


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
	bool Train(const std::vector<MVector> x, const std::vector<MVector> y,
			   double initsd, double learningRate, double costThreshold, int maxIterations)
	{
		// Check that there are the same number of training data inputs as outputs
		assert(x.size() == y.size());

		// TODO: Step 2 - initialise the weights and biases with the standard deviation "initsd"
		
		for (int iter=1; iter<=maxIterations; iter++)
		{
			// Step 3: Choose a random training data point i in {0, 1, 2, ..., N}
			int i = rnd()%x.size();

			// TODO: Step 4 - run the feed-forward algorithm

			// TODO: Step 5 - run the back-propagation algorithm
			
			// TODO: Step 6 - update the weights and biases using stochastic gradient
			//                with learning rate "learningRate"

			// Every so often, perform step 7 and show an update on how the cost function has decreased
			// Here, "every so often" means once every 1000 iterations, and also at the last iteration
			if ((!(iter%1000)) || iter==maxIterations)
			{
				// TODO: Step 7(a) - calculate the total cost

				// TODO: display the iteration number and total cost to the screen
				
				// TODO: Step 7(b) - return from this method with a value of true,
	 			//                   indicating success, if this cost is less than "costThreshold".
			}
			
		} // Step 8: go back to step 3, until we have taken "maxIterations" steps

		// Step 9: return "false", indicating that the training did not succeed.
		return false;
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
		// TODO: Return d/dz(sigma(z))
		return (1.0 / pow(cosh(z), 2));
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

		// TODO: Loop over all components of all the weight matrices
		//       and bias vectors at each relevant layer of the network.
	}

	// Evaluate the feed-forward algorithm, setting weighted inputs and activations
	// at each layer, given an input vector x
	void FeedForward(const MVector &x)
	{
		// Check that the input vector has the same number of elements as the input layer
		assert(x.size() == nneurons[0]);
		
		// TODO: Implement the feed-forward algorithm, equations (1.7), (1.8)
	}

	// Evaluate the back-propagation algorithm, setting errors for each layer 
	void BackPropagateError(const MVector &y)
	{
		// Check that the output vector y has the same number of elements as the output layer
		assert(y.size() == nneurons[nLayers - 1]);

		// TODO: Implement the back-propagation algorithm, equations (1.22) and (1.24)
	}

	
	// Apply one iteration of the stochastic gradient iteration with learning rate eta.
	void UpdateWeightsAndBiases(double eta)
	{
		// Check that the learning rate is positive
		assert(eta>0);
		
		// TODO: update the weights and biases according to the stochastic gradient
		//       iteration, using equations (1.25) and (1.26) to evaluate
		//       the components of grad C.
	}

	
	// Return the cost function of the network with respect to a single the desired output y
	// Note: call FeedForward(x) first to evaluate the network output for an input x,
	//       then call this method Cost(y) with the corresponding desired output y
	double Cost(const MVector &y)
	{
		// Check that y has the same number of elements as the network has outputs
		assert(y.size() == nneurons[nLayers-1]);
		
		// TODO: Return the cost associated with this output
	}

	// Return the total cost C for a set of training data x and desired outputs y
	double TotalCost(const std::vector<MVector> x, const std::vector<MVector> y)
	{
		// Check that there are the same number of inputs as outputs
		assert(x.size() == y.size());

		// TODO: Implement the cost function, equation (1.9), using
		//       the FeedForward(x) and Cost(y) methods
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
void ClassifyTestData()
{
	// Create a network with two input neurons, two hidden layers of three neurons, and one output neuron
	Network n({2, 3, 3, 1});

	// Get some data to train the network
	std::vector<MVector> x, y;
	GetTestData(x, y);
	
	// Train network on training inputs x and outputs y
	// Numerical parameters are:
	//  initial weight and bias standard deviation = 0.1
	//  learning rate = 0.1
	//  cost threshold = 1e-4
	//  maximum number of iterations = 10000
	bool trainingSucceeded = n.Train(x, y, 0.1, 0.1, 1e-4, 1000000);

	// If training failed, report this
	if (!trainingSucceeded)
	{
		std::cout << "Failed to converge to desired tolerance." << std::endl;
	}
	
	// Generate some output files for plotting
	ExportTrainingData(x, y, "test_points.txt");
	n.ExportOutput("test_contour.txt");
}


int main()
{
	// Call the test function	
	bool testsPassed = Network::Test(3);

	// If tests did not pass, something is wrong; end program now
	if (!testsPassed)
	{
		std::cout << "A test failed." << std::endl;
		return 1;
	}

	std::cout << "Tests passed, procede to example program...\n" << std::endl;

	// Tests passed, so run our example program.
	ClassifyTestData();

	return 0;
}

