import unittest
import torch




class TestFeatureSize(unittest.TestCase):
    
    def __init__(self, model, expected_feature_size):
        super().__init__()  # Call the original constructor of TestCase
        self.model = model
        self.expected_feature_size = expected_feature_size
    
    # def test_feature_size(self):
    def test_feature_size(self):

        # Switch to feature extraction mode
        self.model.feature_extractor_mode()
        
        # Create a dummy input tensor
        x = torch.randn(1, 3, 224, 224)  # Assuming the input size is (3, 224, 224)==> [batch_size, channels, height, width]
        
        # Forward pass
        with torch.inference_mode():  # Use torch.no_grad() if torch.inference_mode() is not available in your PyTorch version
            features = self.model(x)
        
        # Check the feature size
        # assert features.shape[1] == self.expected_feature_size, f"Expected feature size {self.expected_feature_size}, got {features.shape[1]}"
         # Check the feature size
        self.assertEqual(features.shape[1], self.expected_feature_size, f"Expected feature size {self.expected_feature_size}, got {features.shape[1]}")
        
        return True
