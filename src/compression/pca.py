

# TODO: create another pca class purely for visual purpsees (Swap's idea). You basically need to take the features from each model (after training) then run PCA on a subset of the 200 labels (for example, 20 labels) and 
# then plot the two component (on xy axis) and you should see some clustering. You can also do this for the entire 200 labels but it will be harder to visualize.
import numpy as np
from sklearn.decomposition import PCA

class PCAWrapper:
# TODO: ADD torch to numpy conversion sanity check

# Question to answer:
# 1. Do I compress the query vector only and then measure distance with the gallery?
# 2. Do I compress the features to the size I want (for example 8 bit) or do I compress the features
#    to a size that's a little bit smaller, say 256 and then take the first 8 components of that? which approach is better?
# 3. in all cases does the gallery features also needs to be compressed? If so, how I extract the original image from the gallery so I can view it?    

# ADD torch to numpy conversion sanity check 


    def __init__(self, n_components=8):
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components)
        self.is_fitted = False

    def fit(self, feature_vector):
        self.pca.fit(feature_vector)
        self.is_fitted = True

    def transform(self, feature_vector):
        if not self.is_fitted:
            raise Exception("PCA not fitted. Call fit() first.")
        return self.pca.transform(feature_vector)

    def compress(self, feature_vector):
        if not self.is_fitted:
            self.fit(feature_vector)
        return self.transform(feature_vector)




# # example usage:
# n_components = 2048  # total components in your feature vector
# num_components_to_keep = 500  # for example, keep 500 components after compression

# pca_wrapper = PCAWrapper(n_components=2048, num_components_to_keep=500)
# train_features_reduced = pca_wrapper.compress(train_features)
# test_features_reduced = pca_wrapper.transform(test_features)




