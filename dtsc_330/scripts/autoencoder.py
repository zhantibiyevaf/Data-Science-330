# Keras is a fancy way to define layers on top of tensorflow
import keras
import numpy as np
from keras.losses import mean_squared_error

class Autoencoder():
    def __init__(self, n_inputs: int, n_constriction: int, n_layers: int):
        self.n_inputs = n_inputs
        self.n_constriction = n_constriction
        self.n_layers = n_layers
        self.encoder, self.full = self._create(n_inputs, n_constriction, n_layers)

    def train(self, features):
        # Epochs = the number of times we loop through the entire dataset
        self.full.compile(loss=mean_squared_error, optimizer='adam')
        self.full.fit(features.values, 
                      features.values,
                      epochs=50,
                      batch_size=32,
                      shuffle=True)

    def embed(self, features):
        pass

    def _create(self, n_inputs, n_constriction, n_layers):
        # Always create inputs layer at the top and name it something special
        # Add outputs layer at the bottom
        # Loop through the inners recursively
            # (recursive iterates on itself)
        # A tensor is kind of like a vector or matrix
        # It IS a vector or matrix
        
        inputs = keras.layers.Input(shape=(n_inputs, ))
        # A trailing comma within parentheses makes it a tuple of length 1
        # A tuple of length 2 is (a, b)

        layers = np.linspace(n_inputs, n_constriction, (n_layers - 1)//2).astype(int)

        x = inputs
        for layer_size in layers[1:-1]:
            x = keras.layers.Dense(int(layer_size), activation='relu')(x)

        constriction_layer = keras.layers.Dense(n_constriction, activation="relu")(x)
        x = constriction_layer

        for layer_size in layers[1:-1][::-1]:
            x = keras.layers.Dense(int(layer_size), activation='relu')(x)

        outputs = keras.layers.Dense(n_inputs, activation="relu")(x)
        # If we created a classifier and we were prediction probability, activation="sigmoid"
        # If we are creating a regression, activation="linear"

        encoder_model = keras.models.Model(inputs=inputs, outputs=constriction_layer)
        decoder_model = keras.models.Model(inputs=constriction_layer, outputs=outputs)
        full_model = keras.models.Model(inputs=inputs, outputs=outputs)

        return encoder_model, full_model
    

if __name__ == '__main__':
    import pandas as pd
    import zipfile

    zf = zipfile.ZipFile('data/wine_quality.zip')
    df = pd.read_csv(zf.open('winequality-white.csv'), sep=';')

    features = df.drop(columns=['quality'])
    labels = features.copy()

    model = Autoencoder(n_inputs=len(features.columns), n_constriction=3, n_layers=7)
    model.train(features)

# Let's consider 7 layers

# Input    -----------
# Hidden 1   -------
# Hidden 2     ---
# Embedding     -
# Hidden 3     ---
# Hidden 4   -------
# Output   -----------
