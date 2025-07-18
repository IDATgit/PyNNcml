import torch
import unittest
import sys
import os

# Adjust path to import model
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from examples.physics_aware_network.models.physics_informed_rain_estimation import PhysicsInformedRainEstimation

class TestPhysicsInformedRainEstimation(unittest.TestCase):
    def test_forward_pass(self):
        """
        Tests the forward pass of the PhysicsInformedRainEstimation model.
        """
        # Model parameters
        metadata_features = 2
        style_vector_dim = 128

        # Create a model instance
        model = PhysicsInformedRainEstimation(
            metadata_features_baseline=metadata_features,
            style_vector_dim_baseline=style_vector_dim,
            metadata_features_rain_rate=metadata_features,
            style_vector_dim_rain_rate=style_vector_dim
        )

        # Create dummy input tensors
        batch_size = 4
        sequence_length = 96
        attenuation_input = torch.randn(batch_size, 1, sequence_length)
        metadata_input = torch.randn(batch_size, metadata_features)

        # Perform a forward pass
        estimated_rain_rate, compensated_attenuation = model(attenuation_input, metadata_input)

        # 1. Check output shapes
        self.assertEqual(estimated_rain_rate.shape, attenuation_input.shape,
                         "Estimated rain rate shape should match input shape")
        self.assertEqual(compensated_attenuation.shape, attenuation_input.shape,
                         "Compensated attenuation shape should match input shape")

        # 2. Check for non-negative outputs
        self.assertTrue(torch.all(estimated_rain_rate >= 0),
                        "Estimated rain rate values should be non-negative")
        self.assertTrue(torch.all(compensated_attenuation >= 0),
                        "Compensated attenuation values should be non-negative")

if __name__ == '__main__':
    unittest.main() 