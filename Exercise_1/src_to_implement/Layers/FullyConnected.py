import numpy as np

from Layers.Base import BaseLayer

class FullyConnected(BaseLayer): # fully connected layer, also known as dense layer
    def __init__(self, input_size, output_size): #initialize the layer with input and output sizes
        super().__init__()
        self.trainable = True #to indicate that this layer has trainable parameters
        self.weights = np.random.uniform(0, 1, (input_size+1, output_size)) # +1 for bias
        # self.bias = np.random.uniform(0, 1, (output_size,))
        self.input_tensor = None # stores input tensor for forward pass as it is needed for backward pass
        self._optimizer = None  # useful for updating weights
        self._gradient_weights = None  # Placeholder for gradient

    # Forward Pass
    def forward(self, input_tensor):
        bias = np.ones((input_tensor.shape[0], 1)) # bias term for each input sample as we need to account for bias
        self.input_tensor = np.concatenate((input_tensor, bias), axis=1) # concatenate bias to input tensor
        return np.dot(self.input_tensor, self.weights)

    # Getter for optimizer
    @property # used to encapsulate the optimizer
    def optimizer(self):
        return self._optimizer

    # Setter for Optimizer
    @optimizer.setter # setting the optimizer for this layer this helps to update the weights
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    # Backward Pass
    def backward(self, error_tensor):
        # Gradient w.r.t weights and bias
        self._gradient_weights = np.dot(self.input_tensor.T, error_tensor) # we calculate the gradient of the loss with respect to the weights

        # weight update if optimizer is set
        if self._optimizer is not None: # check if optimizer is set
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
        # Return error for previous layer (excluding bias gradient)
        return np.dot(error_tensor, self.weights[:-1, :].T)

    @property
    def gradient_weights(self): # this is used to get the gradient of the weights
        return self._gradient_weights
