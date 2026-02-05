import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging

logger = logging.getLogger(__name__)

class VoiceClassifier:

    def __init__(self, model_path: str = "models/weights"):
        self.model_path = model_path
        self.scaler = StandardScaler()
        self.classifier = None
        self.feature_names = None
        self._initialize_model()

    def _initialize_model(self):
        model_file = os.path.join(self.model_path, "classifier.pkl")
        scaler_file = os.path.join(self.model_path, "scaler.pkl")

        if os.path.exists(model_file) and os.path.exists(scaler_file):
            logger.info("Loading pre-trained model")
            self.classifier = joblib.load(model_file)
            self.scaler = joblib.load(scaler_file)
        else:
            logger.info("Initializing new model (demo mode)")
            self.classifier = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            self._create_demo_model()

    def _create_demo_model(self):
        logger.info("Creating demo model with synthetic training data")

        n_samples = 200
        n_features = 200

        ai_features = np.random.randn(n_samples, n_features) * 0.5
        ai_features[:, :40] += np.random.uniform(0.3, 0.5, (n_samples, 40))
        ai_features[:, 160:165] *= 0.3

        human_features = np.random.randn(n_samples, n_features) * 1.2
        human_features[:, :40] += np.random.uniform(-0.2, 0.3, (n_samples, 40))
        human_features[:, 160:165] *= 2.0

        X = np.vstack([ai_features, human_features])
        y = np.array([0] * n_samples + [1] * n_samples)

        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]

        X_scaled = self.scaler.fit_transform(X)
        self.classifier.fit(X_scaled, y)

        logger.info(f"Demo model trained with accuracy: {self.classifier.score(X_scaled, y):.3f}")

    def predict(self, features: dict) -> tuple[str, float]:
        try:
            feature_array = self._features_dict_to_array(features)

            if feature_array.ndim == 1:
                feature_array = feature_array.reshape(1, -1)

            expected_size = 190
            if feature_array.shape[1] < expected_size:
                padding = np.zeros((1, expected_size - feature_array.shape[1]))
                feature_array = np.hstack([feature_array, padding])
            elif feature_array.shape[1] > expected_size:
                feature_array = feature_array[:, :expected_size]

            feature_scaled = self.scaler.transform(feature_array)

            feature_scaled = np.nan_to_num(feature_scaled, nan=0.0, posinf=0.0, neginf=0.0)

            prediction = self.classifier.predict(feature_scaled)[0]
            probabilities = self.classifier.predict_proba(feature_scaled)[0]

            confidence = probabilities[prediction]

            classification = "HUMAN" if prediction == 1 else "AI_GENERATED"

            classification, confidence = self._apply_heuristics(
                features, classification, confidence
            )

            return classification, float(confidence)

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return "HUMAN", 0.5

    def _features_dict_to_array(self, features: dict) -> np.ndarray:
        sorted_keys = sorted(features.keys())
        return np.array([features[k] for k in sorted_keys], dtype=np.float32)

    def _apply_heuristics(self, features: dict, classification: str, confidence: float) -> tuple[str, float]:
        jitter = features.get('jitter', 0.05)
        if jitter < 0.008:
            if classification == "HUMAN":
                confidence *= 0.8
            else:
                confidence = min(confidence * 1.1, 0.99)

        shimmer = features.get('shimmer', 0.05)
        if shimmer < 0.015:
            if classification == "HUMAN":
                confidence *= 0.85
            else:
                confidence = min(confidence * 1.1, 0.99)

        f0_std = features.get('f0_std', 10)
        if f0_std < 5:
            if classification == "HUMAN":
                confidence *= 0.9
            else:
                confidence = min(confidence * 1.05, 0.99)

        flatness_std = features.get('spectral_flatness_std', 0.1)
        if flatness_std < 0.05:
            if classification == "HUMAN":
                confidence *= 0.9

        return classification, confidence

    def is_loaded(self) -> bool:
        return self.classifier is not None

    def save_model(self):
        os.makedirs(self.model_path, exist_ok=True)
        joblib.dump(self.classifier, os.path.join(self.model_path, "classifier.pkl"))
        joblib.dump(self.scaler, os.path.join(self.model_path, "scaler.pkl"))
        logger.info(f"Model saved to {self.model_path}")