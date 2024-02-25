# Neural Network for XOR Learning and Image Recognition

This project is a simple implementation of a neural network capable of learning the XOR logical function as well as recognizing digits in images.

## Description

The neural network is implemented in C++ and initially designed to learn the XOR logical function. It includes functionality for recognizing digits in images using the MNIST dataset.

The goal is to demonstrate the fundamental operations of a neural network and its ability to solve both classification and image recognition problems.

## Usage

The neural network's architecture is customizable in terms of depth, width, and other parameters. You can adjust these settings in the `AI.h` file to customize the network according to your requirements.

### Execution

To execute the program, follow these steps:

```
$ make
$ ./main XOR      // Trains the network for XOR learning
$ ./main UXOR     // Retrains the network for XOR learning
$ ./main MNIST    // Retrains the network from scratch for image recognition using the MNIST dataset
$ ./main UMNIST   // Retrains the network for image recognition using the MNIST dataset
$ ./main TXOR     // Tests the trained neural network with XOR values
$ ./main TMNIST1  // Tests the trained neural network with the MNIST training dataset
$ ./main TMNIST2  // Tests the trained neural network with a custom image located in the img/test_img.png directory
$ make clean
```

You can directly test the network after training by appending the testing command. For example:

```
$ ./main UMNIST TMNIST2
```

The network will be retrained for XOR learning, and then it will be tested with a custom image.

The weight files generated during training are stored in the `train/` directory.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.