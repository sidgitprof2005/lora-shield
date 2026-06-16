"""
LoRA Shield - Feature Extractor

This module implements the LoRAFeatureExtractor class to extract SVD spectral and
statistical features from LoRA weight matrices to detect poisoned adapters.
"""

import os
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
from safetensors import safe_open
from scipy.stats import kurtosis, skew

class LoRAFeatureExtractor:
    def __init__(self):
        """
        Initialize the LoRA Feature Extractor.
        """
        pass

    def load_adapter_weights(self, adapter_path: str) -> Dict[str, torch.Tensor]:
        """
        Loads the adapter weight matrices from the given directory.
        Supports safetensors format and falls back to pytorch .bin format.
        """
        safetensors_path = os.path.join(adapter_path, "adapter_model.safetensors")
        bin_path = os.path.join(adapter_path, "adapter_model.bin")
        
        weights = {}
        if os.path.exists(safetensors_path):
            try:
                with safe_open(safetensors_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        weights[key] = f.get_tensor(key)
            except Exception as e:
                print(f"Error loading safetensors from {safetensors_path}: {e}")
                
        if not weights and os.path.exists(bin_path):
            try:
                weights = torch.load(bin_path, map_location="cpu")
            except Exception as e:
                print(f"Error loading bin file from {bin_path}: {e}")
                
        if not weights:
            raise FileNotFoundError(f"No valid adapter weight files found at {adapter_path}")
            
        return weights

    def _pair_lora_matrices(self, weights: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Pairs matching lora_A and lora_B weight matrices for each target layer.
        """
        pairs = {}
        for name, tensor in weights.items():
            # Standard PEFT names contain: ...lora_A.weight or ...lora_B.weight
            if "lora_A" in name:
                # Resolve base parameter name key (e.g. attention query or value projection)
                base_key = name.split(".lora_A")[0]
                if base_key not in pairs:
                    pairs[base_key] = {}
                pairs[base_key]["lora_A"] = tensor
            elif "lora_B" in name:
                base_key = name.split(".lora_B")[0]
                if base_key not in pairs:
                    pairs[base_key] = {}
                pairs[base_key]["lora_B"] = tensor
        return pairs

    def extract_features(self, adapter_path: str) -> np.ndarray:
        """
        Extracts 10-dimensional features from a single LoRA adapter checkpoint.
        
        Returns:
            A 1D numpy array of shape (10,) representing the averaged features.
        """
        print(f"Extracting features from adapter: {adapter_path}")
        try:
            weights = self.load_adapter_weights(adapter_path)
        except Exception as e:
            print(f"Failed to load weights for {adapter_path}: {e}")
            return np.zeros(10)
            
        pairs = self._pair_lora_matrices(weights)
        
        layer_features = []
        for base_key, pair in pairs.items():
            lora_A = pair.get("lora_A")
            lora_B = pair.get("lora_B")
            
            if lora_A is None or lora_B is None:
                continue
                
            # Move to CPU numpy
            A = lora_A.detach().cpu().float().numpy()
            B = lora_B.detach().cpu().float().numpy()
            
            # Reconstruct lora weight delta delta_W = B @ A
            # If A has shape (r, in_features) and B has shape (out_features, r)
            # then product B @ A is of shape (out_features, in_features).
            # If shapes are transposed in some variations, transpose them
            if A.ndim == 2 and B.ndim == 2:
                if A.shape[0] != B.shape[1]:
                    # Attempt automated alignment transposes if needed
                    if A.shape[1] == B.shape[0]:
                        delta_W = A.T @ B.T
                    elif A.shape[0] == B.shape[0]:
                        delta_W = B.T @ A
                    else:
                        delta_W = B @ A.T
                else:
                    delta_W = B @ A
            else:
                # If 1D bias or other mismatch
                continue
                
            # --- 1. Spectral Features (using SVD) ---
            try:
                # SVD of delta_W
                S = np.linalg.svd(delta_W, compute_uv=False)
            except Exception as e:
                print(f"SVD failed for layer {base_key}: {e}")
                continue
                
            if len(S) == 0:
                continue
                
            # Spectral features
            mean_singular = float(np.mean(S))
            std_singular = float(np.std(S))
            max_singular = float(np.max(S))
            
            sum_singular = float(np.sum(S))
            spectral_concentration = float(S[0] / sum_singular) if sum_singular > 1e-9 else 0.0
            
            # Spectral entropy
            p = S / (sum_singular + 1e-12)
            # Use small epsilon to avoid log(0)
            spectral_entropy = float(-np.sum(p * np.log(p + 1e-12)))
            
            # --- 2. Statistical Features ---
            w_flat = delta_W.flatten()
            w_mean = float(np.mean(w_flat))
            w_std = float(np.std(w_flat))
            w_max_abs = float(np.max(np.abs(w_flat)))
            w_kurtosis = float(kurtosis(w_flat, fisher=True, bias=True))
            w_skewness = float(skew(w_flat, bias=True))
            
            features = [
                # Spectral
                mean_singular,
                std_singular,
                max_singular,
                spectral_concentration,
                spectral_entropy,
                # Statistical
                w_mean,
                w_std,
                w_max_abs,
                w_kurtosis,
                w_skewness
            ]
            layer_features.append(features)
            
        if not layer_features:
            print(f"Warning: No valid LoRA layer pairs found in {adapter_path}")
            return np.zeros(10)
            
        # Average across all layers
        avg_features = np.mean(layer_features, axis=0)
        return avg_features

    def batch_extract(self, folder_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Processes an entire folder of adapters.
        Iterates over subfolders, extracts features, and determines label.
        Label: 1 if "poisoned" is in the adapter folder path, else 0 (clean).
        
        Returns:
            X: Feature matrix of shape (num_adapters, 10)
            y: Label array of shape (num_adapters,)
        """
        print(f"\n--- Batch extracting from folder: {folder_path} ---")
        
        X_list = []
        y_list = []
        
        # Walk to find directories containing adapter configs
        for root, dirs, files in os.walk(folder_path):
            if "adapter_config.json" in files:
                adapter_name = os.path.basename(root)
                print(f"Found adapter: {adapter_name}")
                
                # Determine label
                if "poisoned" in root or "poison" in root or "poisoned" in adapter_name or "poison" in adapter_name:
                    label = 1
                else:
                    label = 0
                    
                features = self.extract_features(root)
                X_list.append(features)
                y_list.append(label)
                print(f"Extracted features: {features} | Label: {label}")
                
        if not X_list:
            print("No adapters found in batch search path.")
            return np.empty((0, 10)), np.empty((0,))
            
        return np.array(X_list), np.array(y_list)

if __name__ == "__main__":
    # Self-test using mock folder structures if run directly
    print("Testing LoRAFeatureExtractor interface...")
    extractor = LoRAFeatureExtractor()
    mock_weights = {
        "model.layers.0.q_proj.lora_A.weight": torch.randn(8, 768),
        "model.layers.0.q_proj.lora_B.weight": torch.randn(768, 8)
    }
    
    # Save mock weights locally for testing
    import tempfile
    from safetensors.torch import save_file
    
    with tempfile.TemporaryDirectory() as tmpdir:
        save_file(mock_weights, os.path.join(tmpdir, "adapter_model.safetensors"))
        with open(os.path.join(tmpdir, "adapter_config.json"), "w") as f:
            f.write('{"base_model_name_or_path": "mock"}')
            
        features = extractor.extract_features(tmpdir)
        print("Self-test extracted features:", features)
        assert features.shape == (10,), f"Expected shape (10,), got {features.shape}"
        print("Extraction verification passed.")
