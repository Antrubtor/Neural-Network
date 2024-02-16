# Neural Network for XOR Learning

This project is a simple implementation of a neural network capable of learning the XOR logical function.

## Description

The neural network is implemented in C++ and uses a basic approach for XOR learning. The goal is to demonstrate the fundamental operation of a neural network and its use in solving a simple classification problem.

The neural network is completely customizable in terms of depth and width. By modifying the `AI.h` file, you can adjust various aspects of the network architecture including the number of layers, the number of neurons in each layer and learning rate.

## Usage

To modify the structure of the neural network, you need to edit the `AI.h` file. In this file, you can adjust the network parameters..

## Example Execution

Here's an example of how to run the program:

```
$ make
$ ./main XOR
```

The program will train the neural network to learn the XOR function. Once the training is completed, it will test the network with different inputs to verify its accuracy.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Future Development

In the future, the neural network will be extended to recognize digits in images. Stay tuned for updates!