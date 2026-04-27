"""
Machine Learning model for flood prediction validation using Real vs Simulated data.
Validates simulation accuracy against real flood events using Random Forest.

Reference:
    Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
    Murphy et al. (2012). Validating flood risk models. Water Resources Research.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.model_selection import train_test_split, cross_val_score  # type: ignore
from sklearn.metrics import (  # type: ignore
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
from typing import Tuple, Dict, Optional, cast
import logging

logger = logging.getLogger(__name__)


class FloodValidationModel:
    """
    Random Forest model for validating flood simulations against real observations.
    
    This model answers: "Can we predict real flood occurrence using:
    - DEM features (elevation, slope)
    - Rainfall data (INMET)
    - Simulation output (simulated flood)"
    """
    
    def __init__(self, 
                 n_estimators: int = 100,
                 random_state: int = 42,
                 test_size: float = 0.2):
        """
        Initialize flood validation model.
        
        Parameters
        ----------
        n_estimators : int
            Number of trees in Random Forest (Breiman 2001)
        random_state : int
            Random seed for reproducibility
        test_size : float
            Fraction of data to use for testing
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.test_size = test_size
        
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.metrics = None
        
    def prepare_data(self, df: pd.DataFrame, 
                    feature_cols: Optional[list] = None,
                    target_col: str = 'real_flood') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare features and target.
        
        Parameters
        ----------
        df : pd.DataFrame
            Dataset with features and target
        feature_cols : list, optional
            Columns to use as features. If None, uses all except target.
        target_col : str
            Target column name
            
        Returns
        -------
        tuple
            (X_train, X_test, y_train, y_test)
        """
        
        if feature_cols is None:
            feature_cols = [col for col in df.columns if col != target_col]
        
        self.feature_names = feature_cols
        
        X = df[feature_cols].values
        y = df[target_col].values
        
        # Split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y.astype(int) if isinstance(y, (pd.Series, np.ndarray)) else y  # type: ignore # Maintain class distribution
        )
        
        logger.info(f"✅ Data prepared:")
        logger.info(f"   Training: {len(self.X_train)} samples")
        logger.info(f"   Testing: {len(self.X_test)} samples")
        logger.info(f"   Features: {len(feature_cols)}")
        
        # Use numpy operations for both array and Series
        pos_count = np.sum(y == 1)
        pos_pct = 100 * pos_count / len(y)
        logger.info(f"   Positive class: {pos_count} ({pos_pct:.1f}%)")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train(self) -> 'FloodValidationModel':
        """
        Train Random Forest model.
        
        Returns
        -------
        self
            Fitted model
        """
        if self.X_train is None:
            raise ValueError("Call prepare_data() first")
        
        logger.info(f"🚀 Training Random Forest ({self.n_estimators} trees)...")
        
        # Clean any NaN/Inf in training data before fitting
        X_train_clean = np.nan_to_num(
            cast(np.ndarray, self.X_train),
            nan=0.0,
            posinf=1.0,
            neginf=0.0
        )
        X_train_clean = np.clip(X_train_clean, 0, 1)
        
        n_nan_before = np.isnan(cast(np.ndarray, self.X_train)).sum()
        if n_nan_before > 0:
            logger.warning(f"Cleaned {n_nan_before} NaN/Inf values from training data")
        
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight='balanced'  # Handle class imbalance
        )
        
        self.model.fit(X_train_clean, cast(np.ndarray, self.y_train))  # type: ignore
        
        logger.info(f"✅ Model trained successfully")
        
        return self
    
    def evaluate(self, verbose: bool = True) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Parameters
        ----------
        verbose : bool
            Print detailed report
            
        Returns
        -------
        dict
            Metrics: accuracy, precision, recall, f1, roc_auc
        """
        if self.model is None:
            raise ValueError("Call train() first")
        
        # Predictions - clean data first
        X_test_arr = cast(np.ndarray, self.X_test)
        y_test_arr = cast(np.ndarray, self.y_test)
        
        # Clean any NaN/Inf in test data before prediction
        X_test_clean = np.nan_to_num(
            X_test_arr,
            nan=0.0,
            posinf=1.0,
            neginf=0.0
        )
        X_test_clean = np.clip(X_test_clean, 0, 1)
        
        y_pred = self.model.predict(X_test_clean)  # type: ignore
        y_proba = self.model.predict_proba(X_test_clean)[:, 1]  # type: ignore
        
        # Compute metrics
        self.metrics = {
            'accuracy': accuracy_score(y_test_arr, y_pred),  # type: ignore
            'precision': precision_score(y_test_arr, y_pred, zero_division=0),  # type: ignore
            'recall': recall_score(y_test_arr, y_pred, zero_division=0),  # type: ignore
            'f1': f1_score(y_test_arr, y_pred, zero_division=0),  # type: ignore
            'roc_auc': roc_auc_score(y_test_arr, y_proba),  # type: ignore
        }
        
        cm = confusion_matrix(y_test_arr, y_pred)  # type: ignore
        self.metrics['tn'] = cm[0, 0]
        self.metrics['fp'] = cm[0, 1]
        self.metrics['fn'] = cm[1, 0]
        self.metrics['tp'] = cm[1, 1]
        
        if verbose:
            self._print_evaluation_report()
        
        return self.metrics
    
    def cross_validate(self, cv: int = 5) -> Dict[str, np.ndarray]:
        """
        Perform cross-validation.
        
        Parameters
        ----------
        cv : int
            Number of folds
            
        Returns
        -------
        dict
            Cross-validation scores
        """
        if self.X_train is None:
            raise ValueError("Call prepare_data() first")
        
        logger.info(f"🔄 Running {cv}-fold cross-validation...")
        
        model_safe = cast(RandomForestClassifier, self.model)
        X_train_arr = cast(np.ndarray, self.X_train)
        y_train_arr = cast(np.ndarray, self.y_train)
        
        scores = {
            'accuracy': cross_val_score(model_safe, X_train_arr, y_train_arr,  # type: ignore
                                       cv=cv, scoring='accuracy'),
            'f1': cross_val_score(model_safe, X_train_arr, y_train_arr,  # type: ignore
                                 cv=cv, scoring='f1'),
            'roc_auc': cross_val_score(model_safe, X_train_arr, y_train_arr,  # type: ignore
                                      cv=cv, scoring='roc_auc'),
        }
        
        logger.info(f"✅ Cross-validation results:")
        for metric, values in scores.items():
            logger.info(f"   {metric}: {values.mean():.3f} (+/- {values.std():.3f})")
        
        return scores
    
    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """
        Get feature importance from trained model.
        
        Parameters
        ----------
        top_n : int
            Number of top features to return
            
        Returns
        -------
        pd.DataFrame
            Feature importances sorted by importance
        """
        if self.model is None:
            raise ValueError("Call train() first")
        
        importances = self.model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def predict_flood_map(self, dataset: pd.DataFrame) -> np.ndarray:
        """
        Predict flood map using trained model.
        
        Parameters
        ----------
        dataset : pd.DataFrame
            Dataset with features (in same format as training)
            
        Returns
        -------
        np.ndarray
            Predicted probabilities (0-1)
        """
        if self.model is None:
            raise ValueError("Call train() first")
        
        X = dataset[self.feature_names].values
        model_safe = cast(RandomForestClassifier, self.model)
        return model_safe.predict_proba(X)[:, 1]  # type: ignore
    
    def _print_evaluation_report(self):
        """Print detailed evaluation report."""
        m = self.metrics
        
        if m is None:
            logger.warning("No metrics available. Call evaluate() first.")
            return
        
        print("\n" + "="*60)
        print("FLOOD VALIDATION MODEL - EVALUATION REPORT")
        print("="*60)
        print(f"\n📊 CLASSIFICATION METRICS:")
        print(f"   Accuracy:  {float(m['accuracy']):.4f}")
        print(f"   Precision: {float(m['precision']):.4f} (of predicted floods, % correct)")
        print(f"   Recall:    {float(m['recall']):.4f} (of actual floods, % detected)")
        print(f"   F1-Score:  {float(m['f1']):.4f} (harmonic mean)")
        print(f"   ROC-AUC:   {float(m['roc_auc']):.4f}")
        
        print(f"\n🔢 CONFUSION MATRIX:")
        print(f"   True Negatives (TN):  {int(m['tn'])}")
        print(f"   False Positives (FP): {int(m['fp'])} (overestimation)")
        print(f"   False Negatives (FN): {int(m['fn'])} (underestimation)")
        print(f"   True Positives (TP):  {int(m['tp'])}")
        
        tn_val = float(m['tn'])
        fp_val = float(m['fp'])
        tp_val = float(m['tp'])
        fn_val = float(m['fn'])
        specificity = tn_val / (tn_val + fp_val) if (tn_val + fp_val) > 0 else 0
        sensitivity = tp_val / (tp_val + fn_val) if (tp_val + fn_val) > 0 else 0
        
        print(f"\n📐 ADDITIONAL METRICS:")
        print(f"   Sensitivity (True Positive Rate): {sensitivity:.4f}")
        print(f"   Specificity (True Negative Rate): {specificity:.4f}")
        print("="*60 + "\n")


def compare_predictions(real_flood: np.ndarray,
                       simulated_flood: np.ndarray,
                       predicted_flood: np.ndarray) -> Dict[str, float]:
    """
    Compare three flood maps: real, simulated, predicted.
    
    Parameters
    ----------
    real_flood : np.ndarray
        Real/observed flood map (HxW, 0/1)
    simulated_flood : np.ndarray
        Simulated flood map (HxW, 0/1)
    predicted_flood : np.ndarray
        ML predicted flood map (HxW, 0-1 probabilities)
        
    Returns
    -------
    dict
        Comparison metrics
    """
    
    # Convert probabilities to binary
    pred_binary = (predicted_flood.flatten() > 0.5).astype(int)
    sim_binary = simulated_flood.flatten().astype(int)
    real_binary = real_flood.flatten().astype(int)
    
    # Metrics: Real vs Simulated
    sim_accuracy = accuracy_score(real_binary, sim_binary)
    sim_f1 = f1_score(real_binary, sim_binary, zero_division=0)
    sim_recall = recall_score(real_binary, sim_binary, zero_division=0)
    
    # Metrics: Real vs Predicted
    pred_accuracy = accuracy_score(real_binary, pred_binary)
    pred_f1 = f1_score(real_binary, pred_binary, zero_division=0)
    pred_recall = recall_score(real_binary, pred_binary, zero_division=0)
    
    results = {
        'simulation_accuracy': sim_accuracy,
        'simulation_f1': sim_f1,
        'simulation_recall': sim_recall,
        'prediction_accuracy': pred_accuracy,
        'prediction_f1': pred_f1,
        'prediction_recall': pred_recall,
        'improvement': pred_accuracy - sim_accuracy,
    }
    
    logger.info(f"\n📊 MODEL COMPARISON (Real Flood vs Others):")
    logger.info(f"   Simulation Accuracy: {sim_accuracy:.4f}")
    logger.info(f"   Prediction Accuracy: {pred_accuracy:.4f}")
    logger.info(f"   ✅ Improvement: {results['improvement']:+.4f}")
    
    return results
