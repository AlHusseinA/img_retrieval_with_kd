

# TODO: create another pca class purely for visual purpsees (Swap's idea). You basically need to take the features from each model (after training) then run PCA on a subset of the 200 labels (for example, 20 labels) and 
# then plot the two component (on xy axis) and you should see some clustering. You can also do this for the entire 200 labels but it will be harder to visualize.
import numpy as np
from sklearn.decomposition import PCA

class PCAWrapper:
    def __init__(self, n_components, compression_level):
        self.n_components = n_components
        self.compression_level = compression_level
        self.pca = PCA(n_components=self.n_components)

    def compress(self, feature_vector):
        # Fit PCA on the feature vector
        self.pca.fit(feature_vector)

        # Calculate the number of components to keep based on the compression level
        num_components_to_keep = int(self.n_components * self.compression_level)

        # Transform the feature vector to the compressed vector
        compressed_vector = self.pca.transform(feature_vector)[:, :num_components_to_keep]

        return compressed_vector
