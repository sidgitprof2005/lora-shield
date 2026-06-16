"""
LoRA Shield - Poisoning Detector Pipeline

This module implements the LoRADetector class and training pipeline using XGBoost.
It extracts SVD and statistical features from adapters, trains a binary classifier, 
evaluates performance, and identifies poisoned layers.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb

from defense.extractor import LoRAFeatureExtractor

class LoRADetector:
    def __init__(self, model_path: str = None):
        """
        Initialize the LoRA Detector.
        
        Args:
            model_path: Optional path to a pre-trained detector.json model file.
        """
        self.model = xgb.XGBClassifier()
        self.feature_names = [
            "Mean Singular", "Std Singular", "Max Singular", "Spectral Conc.", "Spectral Entropy",
            "Weight Mean", "Weight Std", "Max Abs Weight", "Kurtosis", "Skewness"
        ]
        
        if model_path is not None:
            self.load_model(model_path)

    def train_pipeline(self, clean_dir: str, poison_dir: str, save_dir: str = None):
        """
        Runs the full training, evaluation, and saving pipeline.
        """
        print("\n=== Initializing Detector Training Pipeline ===")
        extractor = LoRAFeatureExtractor()
        
        # 1. Build dataset
        X_clean, y_clean = extractor.batch_extract(clean_dir)
        X_poison, y_poison = extractor.batch_extract(poison_dir)
        
        # We need sufficient samples (e.g. at least 5 of each class) for stratified splits
        if len(X_clean) >= 5 and len(X_poison) >= 5:
            X = np.vstack([X_clean, X_poison])
            y = np.concatenate([y_clean, y_poison])
        else:
            print("[Warning] Insufficient adapters found (needed at least 5 clean and 5 poisoned). Generating synthetic dataset for demonstration/testing.")
            # Generate synthetic SVD-like features (10 dimensions)
            np.random.seed(42)
            X_clean = np.random.normal(loc=0.1, scale=0.1, size=(25, 10))
            X_poison = np.random.normal(loc=0.8, scale=0.2, size=(25, 10))
            X = np.vstack([X_clean, X_poison])
            y = np.array([0] * 25 + [1] * 25)
            
        # 2. Print statistics
        print("\n--- Dataset Statistics ---")
        print(f"Total samples: {len(X)}")
        print(f"Clean adapters: {np.sum(y == 0)}")
        print(f"Poisoned adapters: {np.sum(y == 1)}")
        print(f"Feature matrix shape: {X.shape}")
        
        # 3. Stratified Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 4. Train XGBoost model
        print("\nTraining XGBoost Classifier...")
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            eval_metric='logloss',
            early_stopping_rounds=10,
            random_state=42
        )
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # 5. Evaluate Model
        print("\n--- Evaluation Results ---")
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=["Clean", "Poisoned"]))
        
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        try:
            auc = roc_auc_score(y_test, y_prob)
            print(f"ROC-AUC Score: {auc:.4f}")
        except Exception as e:
            print(f"ROC-AUC Score could not be computed: {e}")
            
        # Plot Feature Importance
        self.save_feature_importance_plot(save_dir)
        
        # 6. Save Model
        self.save_model(save_dir)

    def save_feature_importance_plot(self, save_dir: str = None):
        """
        Plots feature importance and saves it to feature_importance.png.
        """
        if save_dir is None:
            save_dir = os.path.dirname(__file__)
            
        importances = self.model.feature_importances_
        indices = np.argsort(importances)
        
        plt.figure(figsize=(10, 6))
        plt.title("XGBoost LoRA Weight Feature Importance")
        plt.barh(range(len(importances)), importances[indices], align="center", color="#852DF4")
        plt.yticks(range(len(importances)), [self.feature_names[i] for i in indices])
        plt.xlabel("Relative Importance")
        plt.tight_layout()
        
        plot_path = os.path.join(save_dir, "feature_importance.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Feature importance plot saved to: {plot_path}")

    def save_model(self, save_dir: str = None):
        """
        Saves XGBoost model to detector.json.
        """
        if save_dir is None:
            save_dir = os.path.dirname(__file__)
            
        model_path = os.path.join(save_dir, "detector.json")
        self.model.save_model(model_path)
        print(f"Trained model saved to: {model_path}")

    def load_model(self, model_path: str):
        """
        Loads the trained model from the given path.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model file found at: {model_path}")
        self.model.load_model(model_path)
        print(f"Successfully loaded detector model from: {model_path}")

    def predict(self, adapter_path: str) -> Tuple[str, float, List[str]]:
        """
        Inspects an adapter directory, makes a verdict, and identifies suspicious layers.
        
        Args:
            adapter_path: Path to the adapter folder containing configurations/weights.
            
        Returns:
            verdict: "CLEAN" or "POISONED"
            confidence: Probability score in range 0-100%
            suspicious_layers: List of names of layers classified as anomalous
        """
        extractor = LoRAFeatureExtractor()
        
        # 1. Load weights and pair layers
        try:
            weights = extractor.load_adapter_weights(adapter_path)
        except Exception as e:
            print(f"Error loading weights for prediction: {e}")
            return "CLEAN", 100.0, []
            
        pairs = extractor._pair_lora_matrices(weights)
        
        suspicious_layers = []
        layer_features = []
        
        # 2. Extract and classify each layer individually
        for base_key, pair in pairs.items():
            lora_A = pair.get("lora_A")
            lora_B = pair.get("lora_B")
            
            if lora_A is None or lora_B is None:
                continue
                
            A = lora_A.detach().cpu().float().numpy()
            B = lora_B.detach().cpu().float().numpy()
            
            # Align shapes
            if A.shape[0] != B.shape[1]:
                if A.shape[1] == B.shape[0]:
                    delta_W = A.T @ B.T
                elif A.shape[0] == B.shape[0]:
                    delta_W = B.T @ A
                else:
                    delta_W = B @ A.T
            else:
                delta_W = B @ A
                
            # SVD
            try:
                S = np.linalg.svd(delta_W, compute_uv=False)
            except Exception:
                continue
                
            if len(S) == 0:
                continue
                
            # Compute SVD and statistical features
            mean_singular = float(np.mean(S))
            std_singular = float(np.std(S))
            max_singular = float(np.max(S))
            sum_singular = float(np.sum(S))
            spectral_concentration = float(S[0] / sum_singular) if sum_singular > 1e-9 else 0.0
            p = S / (sum_singular + 1e-12)
            spectral_entropy = float(-np.sum(p * np.log(p + 1e-12)))
            
            w_flat = delta_W.flatten()
            w_mean = float(np.mean(w_flat))
            w_std = float(np.std(w_flat))
            w_max_abs = float(np.max(np.abs(w_flat)))
            from scipy.stats import kurtosis, skew
            w_kurtosis = float(kurtosis(w_flat, fisher=True, bias=True))
            w_skewness = float(skew(w_flat, bias=True))
            
            feat = np.array([
                mean_singular, std_singular, max_singular, spectral_concentration, spectral_entropy,
                w_mean, w_std, w_max_abs, w_kurtosis, w_skewness
            ])
            layer_features.append(feat)
            
            # Classify layer
            layer_prob = float(self.model.predict_proba(feat.reshape(1, -1))[0, 1])
            if layer_prob > 0.5:
                suspicious_layers.append(base_key)
                
        # 3. Overall Verdict using average features
        if not layer_features:
            return "CLEAN", 100.0, []
            
        avg_features = np.mean(layer_features, axis=0)
        overall_prob = float(self.model.predict_proba(avg_features.reshape(1, -1))[0, 1])
        
        if overall_prob > 0.5:
            verdict = "POISONED"
            confidence = overall_prob * 100.0
        else:
            verdict = "CLEAN"
            confidence = (1.0 - overall_prob) * 100.0
            
        return verdict, confidence, suspicious_layers

def main():
    detector = LoRADetector()
    
    # Train pipeline with standard dataset paths
    clean_dir = "data/clean_adapters"
    poison_dir = "data/poisoned_adapters"
    
    detector.train_pipeline(clean_dir, poison_dir)

if __name__ == "__main__":
    main()
