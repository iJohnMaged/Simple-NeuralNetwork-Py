# Simple-NeuralNetwork-Py

A very simple fully connected Neural Network implementation in Python based on the tutorials and the NN.js library by Danial Shiffman is this [playlist].

If you want some resources to learn about Neural Networks, I suggest checking these playlists and books:
* [Neural networks] By [3Blue1Brown]
* [Neural Networks - The Nature of Code] By [The Coding Train]
* [Make Your Own Neural Network] By Tariq Rashid

# To-Do List

* [x] Implmeneting the basic NN itself with only 1 hidden layer.
* [ ] Replace `matrix.py` with `numpy`
* [ ] Train the library to solve XOR problem
* [ ] Train the library on the MNIST dataset
* [ ] Add support for multiple hidden layers
* [ ] Train the library to play games

# Documentation

* `NeuralNetwork` - The neural network class
    * `predict(input_list)` - Returns the output of the NeuralNetwork
    * `train(input_list, output_list)` - Trains the NeuralNetwork on the given input.

# License

This project is licensed under the terms of the MIT license, see LICENSE.


[playlist]: <https://www.youtube.com/watch?v=XJ7HLz9VYz0&list=PLRqwX-V7Uu6aCibgK1PTWWu9by6XFdCfh>
[Neural networks]: <https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi>
[3Blue1Brown]: <https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi>
[Neural Networks - The Nature of Code]: <https://www.youtube.com/playlist?list=PLRqwX-V7Uu6aCibgK1PTWWu9by6XFdCfh>
[The Coding Train]: <https://www.youtube.com/channel/UCvjgXvBlbQiydffZU7m1_aw>
[Make Your Own Neural Network]: <https://www.amazon.com/Make-Your-Own-Neural-Network-ebook/dp/B01EER4Z4G/ref=as_li_ss_tl?ie=UTF8&qid=1498492463&sr=8-1&keywords=make+your+own+neural+network&linkCode=sl1&tag=natureofcode-20&linkId=0d10fdc485d6452bb7fc2b62ab4ffd31>
