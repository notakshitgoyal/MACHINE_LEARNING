# MACHINE_LEARNING

# Project Description: Sine Function Modeling with Neural Networks

## Overview
This project aims to create a neural network model to approximate the sine function. Neural networks have gained popularity in recent years due to their ability to learn complex patterns from data. In this project, we will build a neural network with multiple hidden layers and train it to predict sine values based on input data. We will use Python and essential libraries such as NumPy and Matplotlib to implement the neural network and visualize the results.

## Project Components

### 1. Data Generation
To train our neural network, we first generate a dataset containing input-output pairs. We randomly sample values from a uniform distribution between 0 and 2π to create the input data. The corresponding output values are calculated as sine values of the input data, with the addition of some random noise to simulate real-world data.

### 2. Neural Network Architecture
Our neural network consists of multiple layers, including input, hidden, and output layers. The architecture is defined as follows:
- Input Layer: 1 neuron (to accept the input data)
- Hidden Layers: 3 hidden layers, each with 10 neurons
- Output Layer: 1 neuron (to predict the sine values)

We use the sigmoid activation function for the hidden layers and the identity function for the output layer.

### 3. Forward Pass
The forward pass function calculates the predictions made by the neural network given the input data. It involves matrix multiplications, bias additions, and activation function applications for each layer.

### 4. Loss Functions
We implement two loss functions:
- Forward Loss: Measures the difference between the predicted and actual outputs. We use mean squared error for this purpose.
- Backward Loss: Computes the gradient of the forward loss, which is needed for backpropagation during training.

### 5. Training Loop
The training loop is responsible for updating the network's weights and biases to minimize the forward loss. We perform the following steps in the loop:
- Forward Pass: Calculate predictions and loss.
- Backward Pass: Compute gradients for weights and biases using backpropagation.
- Update Parameters: Adjust weights and biases using gradient descent.
- Monitor Loss: Keep track of the loss during training to assess model performance.

### 6. Visualization
Throughout the project, we use Matplotlib to create visualizations. We plot the original dataset, the model's predictions, and the loss history to gain insights into the training process and model accuracy.

## Conclusion
This project demonstrates the construction and training of a neural network to model the sine function. By following the forward and backward pass procedures and using gradient descent, the network learns to approximate sine values accurately. The visualizations provide valuable insights into the training progress and the model's ability to capture the underlying sine function.
