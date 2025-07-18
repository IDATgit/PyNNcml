import torch
import unittest
import sys
import os

# Adjust path to import model
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from examples.physics_aware_network.models.attanuation_to_rain_rate import AttenuationToRainRate

class TestAttenuationToRainRate(unittest.TestCase):
    def test_forward_pass(self):
        """
        Tests the forward pass of the AttenuationToRainRate model.
        """
        # Model parameters
        metadata_features = 2
        style_vector_dim = 128  # This is used to calculate total_channels inside, but not a direct arg

        # Create a model instance
        model = AttenuationToRainRate(metadata_features, style_vector_dim)

        # Create dummy input tensors
        batch_size = 4
        sequence_length = 96  # Should be the same as the attenuation input
        # The input to this model is a single value from the sequence
        attenuation_input = torch.randn(batch_size, 1) 
        metadata_input = torch.randn(batch_size, metadata_features)

        # Perform a forward pass
        predicted_rain_rate = model(attenuation_input, metadata_input)

        # 1. Check output shape
        self.assertEqual(predicted_rain_rate.shape, attenuation_input.shape,
                         "Output shape should match input shape")

        # 2. Check for non-negative output
        self.assertTrue(torch.all(predicted_rain_rate >= 0),
                        "Output values should be non-negative")

if __name__ == '__main__':
    unittest.main() 