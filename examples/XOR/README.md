# XOR Problem in Neural Networks

The XOR "Exclusive or" is a classic problem in ANN (Artificial Neural Network) research.
It's the problem of using a neural network to predict the about of the XOr operation over two inputs.
The output of the XOR function is 1 for non-equal inputs and 0 for equal inputs.

### XOR Output
|Input 1|Input 2| Input 1 XOR Input 2|
|:-------:|:-------:|:--------------------:|
|0|0|0|
|0|1|1|
|1|0|1|
|1|1|0|

XOR is a [classification problem](https://en.wikipedia.org/wiki/Statistical_classification) and can be approached using 
supervised machine learning algorithms since we already know the expected answers in advance.

The XOR problem isn't linearly separable, we can't draw a single line to separate 0s from 1s.

![XOR Graph](https://i.imgur.com/Ia2qgIl.png "XOR GRAPH")

With the current implementation of the neural network which consists of one input layer, one hidden layer, one output layer the neural
network can achieve non-linear separation.

![Multilayer perceptron](https://i.stack.imgur.com/j70eM.jpg "Multilayer perceptron")

## Output of the XOR problem

### Using sigmoid activation function

![sigmoid output](https://i.imgur.com/iQ0Goo1.jpg "Sigmoid output")
