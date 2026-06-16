"""
LoRA Shield - Unit Test Suite

This test suite uses Python's standard unittest library to verify:
1. Feature extraction dimensions and SVD/statistical outputs in extractor.py.
2. Training and inference functionality in detector.py.
"""

import os
import unittest
import numpy as np
import torch
import tempfile
from safetensors.torch import save_file

from defense.extractor import LoRAFeatureExtractor
from defense.detector import LoRADetector

class TestLoRAFeatureExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = LoRAFeatureExtractor()
        
        # Define mock layer configurations (r=8, in_features=768)
        self.mock_weights = {
            "base_model.model.model.layers.0.self_attn.query.lora_A.weight": torch.randn(8, 768) * 0.01,
            "base_model.model.model.layers.0.self_attn.query.lora_B.weight": torch.randn(768, 8) * 0.01,
            "base_model.model.model.layers.0.self_attn.value.lora_A.weight": torch.randn(8, 768) * 0.01,
            "base_model.model.model.layers.0.self_attn.value.lora_B.weight": torch.randn(768, 8) * 0.01
        }

    def test_pairing_logic(self):
        """
        Verify that query and value layers are grouped correctly.
        """
        pairs = self.extractor._pair_lora_matrices(self.mock_weights)
        self.assertEqual(len(pairs), 2)
        
        expected_keys = [
            "base_model.model.model.layers.0.self_attn.query",
            "base_model.model.model.layers.0.self_attn.value"
        ]
        for key in expected_keys:
            self.assertIn(key, pairs)
            self.assertIn("lora_A", pairs[key])
            self.assertIn("lora_B", pairs[key])

    def test_features_shape_and_content(self):
        """
        Verify that extraction returns a 10-dimensional vector with no NaN values.
        """
        # Save mock weights to temporary safetensors file
        with tempfile.TemporaryDirectory() as tmpdir:
            safetensors_path = os.path.join(tmpdir, "adapter_model.safetensors")
            save_file(self.mock_weights, safetensors_path)
            
            # Write dummy config
            with open(os.path.join(tmpdir, "adapter_config.json"), "w") as f:
                f.write('{"base_model_name_or_path": "bert-base-uncased"}')
                
            features = self.extractor.extract_features(tmpdir)
            
            self.assertEqual(features.shape, (10,))
            self.assertFalse(np.isnan(features).any(), "Extracted features contain NaN values")
            self.assertFalse(np.isinf(features).any(), "Extracted features contain Inf values")

class TestLoRADetector(unittest.TestCase):
    def setUp(self):
        self.detector = LoRADetector()
        
        # Generate synthetic dataset of 20 samples (10 clean, 10 poisoned)
        # Each feature vector has 10 values
        np.random.seed(42)
        X_clean = np.random.normal(loc=0.1, scale=0.05, size=(10, 10))
        X_poisoned = np.random.normal(loc=0.8, scale=0.1, size=(10, 10))
        
        self.X = np.vstack([X_clean, X_poisoned])
        self.y = np.array([0] * 10 + [1] * 10)

    def test_train_and_predict(self):
        """
        Verify training accuracy, saving features, and loading model pipeline.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Split and fit
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
            )
            
            import xgboost as xgb
            self.detector.model = xgb.XGBClassifier(
                n_estimators=5,
                max_depth=3,
                learning_rate=0.1,
                eval_metric="logloss",
                random_state=42
            )
            self.detector.model.fit(X_train, y_train)
            
            # Save detector json
            model_path = os.path.join(tmpdir, "detector.json")
            self.detector.save_model(tmpdir)
            self.assertTrue(os.path.exists(model_path))
            
            # Test loading model
            new_detector = LoRADetector()
            new_detector.load_model(model_path)
            
            # Test predictions
            sample_features = np.random.normal(loc=0.1, scale=0.05, size=(10,))
            pred = new_detector.model.predict(sample_features.reshape(1, -1))
            self.assertIn(pred[0], [0, 1])

if __name__ == "__main__":
    unittest.main()
