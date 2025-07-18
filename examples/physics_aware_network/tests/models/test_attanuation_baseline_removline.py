import torch
import unittest
import sys
import os

# Adjust path to import model
# This allows the test to be run from the root of the project
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from examples.physics_aware_network.models.attanuation_baseline_removline import AttenuationBaselineRemoval

class TestAttenuationBaselineRemoval(unittest.TestCase):
    def test_forward_pass(self):
        """
        Tests the forward pass of the AttenuationBaselineRemoval model.
        """
        # Model parameters
        metadata_features = 2
        style_vector_dim = 128 
        
        # Create a model instance
        model = AttenuationBaselineRemoval(metadata_features, style_vector_dim)
        
        # Create dummy input tensors
        batch_size = 4
        sequence_length = 96 # Example sequence length, like 15-min intervals over 24 hours
        attenuation_input = torch.randn(batch_size, 1, sequence_length)
        metadata_input = torch.randn(batch_size, metadata_features)
        
        # Perform a forward pass
        compensated_attenuation = model(attenuation_input, metadata_input)
        
        # 1. Check output shape
        self.assertEqual(compensated_attenuation.shape, attenuation_input.shape, 
                         "Output shape should match input shape")
                         
        # 2. Check for non-negative output (due to final ReLU)
        self.assertTrue(torch.all(compensated_attenuation >= 0), 
                        "Output values should be non-negative")

if __name__ == '__main__':
    unittest.main() 