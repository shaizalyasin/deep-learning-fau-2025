import copy

class NeuralNetwork:
    def __init__(self, optimizer):
        self.optimizer = optimizer               # Optimizer to be copied into trainable layers
        self.loss = []                           # Stores loss values per training iteration
        self.layers = []                         # Ordered list of layers
        self.data_layer = None                   # Provides (input_tensor, label_tensor)
        self.loss_layer = None                   # final loss layer (e.g. CrossEntropyLoss)

    def forward(self):
        # Get input and label from the data layer
        input_tensor, self.label_tensor = self.data_layer.next()

        # Forward through all layers
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)

        # Pass through loss layer to get loss value
        return self.loss_layer.forward(input_tensor, self.label_tensor)

    def backward(self):
        # Start backprop from loss layer
        error_tensor = self.loss_layer.backward(self.label_tensor)

        # Propagate back through all layers in reverse
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def append_layer(self, layer):
        # Deep copy optimizer for trainable layers
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)

    def train(self, iterations):
        for _ in range(iterations):
            loss_value = self.forward()
            self.loss.append(loss_value)
            self.backward()

    def test(self, input_tensor):
        # Pass input through all layers only (not loss)
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return input_tensor
