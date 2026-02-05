import argparse
import os
import sys
import glob
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tqdm import tqdm
import logging

from utils.audio_processor import AudioProcessor
from utils.feature_extractor import FeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoiceAuthTrainer:
    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.audio_processor = AudioProcessor()
        self.feature_extractor = FeatureExtractor()
        self.scaler = StandardScaler()
        self.classifier = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def load_dataset(self, languages=None):
        logger.info("Loading dataset...")
        
        X_features = []
        y_labels = []
        
        # Load AI-generated samples (label = 0)
        ai_dir = self.data_dir / "ai_generated"
        if ai_dir.exists():
            ai_files = glob.glob(str(ai_dir / "*.mp3")) + glob.glob(str(ai_dir / "*.wav"))
            logger.info(f"Found {len(ai_files)} AI-generated samples")
            
            for file_path in tqdm(ai_files, desc="Processing AI samples"):
                try:
                    features = self._process_audio_file(file_path, languages)
                    if features is not None:
                        X_features.append(features)
                        y_labels.append(0)  # AI = 0
                except Exception as e:
                    logger.warning(f"Failed to process {file_path}: {e}")
        
        # Load human samples (label = 1)
        human_dir = self.data_dir / "human"
        if human_dir.exists():
            human_files = glob.glob(str(human_dir / "*.mp3")) + glob.glob(str(human_dir / "*.wav"))
            logger.info(f"Found {len(human_files)} human samples")
            
            for file_path in tqdm(human_files, desc="Processing human samples"):
                try:
                    features = self._process_audio_file(file_path, languages)
                    if features is not None:
                        X_features.append(features)
                        y_labels.append(1)  # Human = 1
                except Exception as e:
                    logger.warning(f"Failed to process {file_path}: {e}")
        
        if len(X_features) == 0:
            raise ValueError("No valid samples found in dataset")
        
        # Convert to numpy arrays
        X = self._align_features(X_features)
        y = np.array(y_labels)
        
        logger.info(f"Loaded {len(X)} samples ({np.sum(y==0)} AI, {np.sum(y==1)} human)")
        logger.info(f"Feature dimension: {X.shape[1]}")
        
        return X, y
    
    def _process_audio_file(self, file_path: str, languages=None):
        with open(file_path, 'rb') as f:
            audio_bytes = f.read()

        audio, sr = self.audio_processor.load_audio(audio_bytes)

        try:
            self.audio_processor.validate_audio(audio, sr)
        except ValueError:
            return None
        
        # Detect language
        language = self.audio_processor.detect_language(audio, sr)
        
        # Filter by language if specified
        if languages and language not in languages:
            return None
        
        # Extract features
        features = self.feature_extractor.extract_all_features(audio, sr, language)
        
        return features
    
    def _align_features(self, feature_list):
        all_keys = set()
        for features in feature_list:
            all_keys.update(features.keys())

        sorted_keys = sorted(all_keys)

        X = np.zeros((len(feature_list), len(sorted_keys)))

        for i, features in enumerate(feature_list):
            for j, key in enumerate(sorted_keys):
                X[i, j] = features.get(key, 0.0)

        return X
    
    def train(self, X, y, test_size=0.2, cv_folds=5):
        logger.info("Splitting dataset...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        logger.info(f"Training set: {len(self.X_train)} samples")
        logger.info(f"Test set: {len(self.X_test)} samples")

        logger.info("Scaling features...")

        self.X_train = np.nan_to_num(self.X_train, nan=0.0, posinf=0.0, neginf=0.0)
        self.X_test = np.nan_to_num(self.X_test, nan=0.0, posinf=0.0, neginf=0.0)

        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        # Train Gradient Boosting Classifier
        logger.info("Training Gradient Boosting Classifier...")
        self.classifier = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=20,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=42,
            verbose=1
        )
        
        self.classifier.fit(self.X_train, self.y_train)
        
        # Cross-validation
        logger.info(f"Performing {cv_folds}-fold cross-validation...")
        cv_scores = cross_val_score(
            self.classifier, self.X_train, self.y_train, 
            cv=cv_folds, scoring='accuracy'
        )
        logger.info(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    def evaluate(self):
        logger.info("Evaluating model...")

        y_pred = self.classifier.predict(self.X_test)
        y_pred_proba = self.classifier.predict_proba(self.X_test)[:, 1]

        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        print(classification_report(
            self.y_test, y_pred,
            target_names=['AI_GENERATED', 'HUMAN']
        ))

        cm = confusion_matrix(self.y_test, y_pred)
        self._plot_confusion_matrix(cm)

        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        logger.info(f"ROC AUC Score: {roc_auc:.4f}")
        self._plot_roc_curve(self.y_test, y_pred_proba)

        self._plot_feature_importance()

        return {
            'accuracy': self.classifier.score(self.X_test, self.y_test),
            'roc_auc': roc_auc
        }
    
    def _plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['AI_GENERATED', 'HUMAN'],
            yticklabels=['AI_GENERATED', 'HUMAN']
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300)
        logger.info(f"Saved confusion matrix to {self.output_dir / 'confusion_matrix.png'}")
        plt.close()
    
    def _plot_roc_curve(self, y_true, y_scores):
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = roc_auc_score(y_true, y_scores)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'roc_curve.png', dpi=300)
        logger.info(f"Saved ROC curve to {self.output_dir / 'roc_curve.png'}")
        plt.close()
    
    def _plot_feature_importance(self, top_n=20):
        feature_importance = self.classifier.feature_importances_
        indices = np.argsort(feature_importance)[-top_n:]

        plt.figure(figsize=(10, 8))
        plt.barh(range(top_n), feature_importance[indices])
        plt.yticks(range(top_n), [f'Feature {i}' for i in indices])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importances')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_importance.png', dpi=300)
        logger.info(f"Saved feature importance to {self.output_dir / 'feature_importance.png'}")
        plt.close()
    
    def save_model(self):
        model_path = self.output_dir / "classifier.pkl"
        scaler_path = self.output_dir / "scaler.pkl"
        
        joblib.dump(self.classifier, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Scaler saved to {scaler_path}")
    
    def test_inference(self, audio_path: str):
        logger.info(f"Testing inference on {audio_path}")

        with open(audio_path, 'rb') as f:
            audio_bytes = f.read()

        audio, sr = self.audio_processor.load_audio(audio_bytes)
        language = self.audio_processor.detect_language(audio, sr)
        features = self.feature_extractor.extract_all_features(audio, sr, language)

        feature_array = np.array([features.get(k, 0) for k in sorted(features.keys())])
        feature_array = feature_array.reshape(1, -1)

        if feature_array.shape[1] < self.X_train.shape[1]:
            padding = np.zeros((1, self.X_train.shape[1] - feature_array.shape[1]))
            feature_array = np.hstack([feature_array, padding])
        elif feature_array.shape[1] > self.X_train.shape[1]:
            feature_array = feature_array[:, :self.X_train.shape[1]]

        feature_scaled = self.scaler.transform(feature_array)

        prediction = self.classifier.predict(feature_scaled)[0]
        confidence = self.classifier.predict_proba(feature_scaled)[0]

        label = "HUMAN" if prediction == 1 else "AI_GENERATED"
        conf = confidence[prediction]

        logger.info(f"Prediction: {label}")
        logger.info(f"Confidence: {conf:.4f}")
        logger.info(f"Language: {language}")

        return label, conf

def main():
    parser = argparse.ArgumentParser(description='Train voice authenticity detection model')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing ai_generated/ and human/ subdirectories')
    parser.add_argument('--output_dir', type=str, default='./models/weights',
                        help='Directory to save trained model')
    parser.add_argument('--languages', type=str, nargs='+',
                        help='Filter by languages (e.g., tamil english hindi)')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test set size (default: 0.2)')
    parser.add_argument('--cv_folds', type=int, default=5,
                        help='Number of cross-validation folds (default: 5)')
    parser.add_argument('--test_audio', type=str,
                        help='Path to audio file for testing inference')
    
    args = parser.parse_args()

    trainer = VoiceAuthTrainer(args.data_dir, args.output_dir)

    X, y = trainer.load_dataset(languages=args.languages)

    trainer.train(X, y, test_size=args.test_size, cv_folds=args.cv_folds)

    metrics = trainer.evaluate()

    trainer.save_model()

    if args.test_audio:
        trainer.test_inference(args.test_audio)

    logger.info("Training complete!")
    logger.info(f"Final Test Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Final ROC AUC: {metrics['roc_auc']:.4f}")

if __name__ == "__main__":
    main()