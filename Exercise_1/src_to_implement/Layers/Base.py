

class BaseLayer:
    def __init__(self):
        self.trainable = False
        # Optional Members
        self.weight = None # can be needed for trainable layers like FullyConnectedLayer
        self.input_tensor = None # may be useful during forward pass