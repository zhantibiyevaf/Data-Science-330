import numpy as np

# Sometimes these are called "abstract base classes"
class Layer():
    def forward(self, x: np.ndarray):
        raise NotImplementedError('Have not created forward pass')

    def backward(self):
        raise NotImplementedError('Have not created backward pass')
    

class Linear(Layer):
    def forward(self, x: np.ndarray):
        return x