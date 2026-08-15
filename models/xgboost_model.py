"""
V4 -- XGBoost Model
Binary classifier: P(target hit before stop)
Trained on trade outcome labels from V2's labeler.

The real test (from the roadmap):
  Compare BASELINE (every valid setup) vs ML-FILTERED (P(win) > threshold).
  Test thresholds on train/val data. Apply chosen threshold to held-out test.
"""
import sys
import os
import json
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        classification_report, confusion_matrix
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class XGBoostTradeModel:
    """
    XGBoost binary classifier for trade outcome prediction.
    Input: features at entry time
    Output: P(target hit before stop)
    """

    def __init__(self):
        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost not installed. Run: pip install xgboost")

        self.model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric='logloss',
            random_state=42,
            use_label_encoder=False,
        )
        self.feature_columns = None
        self.is_trained = False

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the XGBoost model.

        Args:
            X_train: DataFrame of features
            y_train: Series of labels (1=TARGET, 0=STOP)
            X_val: Optional validation features
            y_val: Optional validation labels
        """
        self.feature_columns = list(X_train.columns)

        fit_params = {}
        if X_val is not None and y_val is not None:
            fit_params['eval_set'] = [(X_val, y_val)]
            fit_params['verbose'] = False

        self.model.fit(X_train, y_train, **fit_params)
        self.is_trained = True

        # Training metrics
        train_pred = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        print(f"  [XGB] Train accuracy: {train_acc:.3f}")

        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_acc = accuracy_score(y_val, val_pred)
            print(f"  [XGB] Val accuracy:   {val_acc:.3f}")

    def predict_proba(self, X):
        """Return P(win) for each sample."""
        if not self.is_trained:
            raise RuntimeError("Model not trained yet")
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X, threshold=0.5):
        """Binary prediction with custom threshold."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def feature_importance(self):
        """Return feature importance as a sorted DataFrame."""
        if not self.is_trained:
            return None
        importance = self.model.feature_importances_
        fi_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        return fi_df

    def save(self, path):
        """Save model to JSON."""
        self.model.save_model(path)
        # Save feature columns alongside
        meta_path = path.replace('.json', '_meta.json')
        with open(meta_path, 'w') as f:
            json.dump({'feature_columns': self.feature_columns}, f)
        print(f"  [XGB] Model saved to {path}")

    def load(self, path):
        """Load model from JSON."""
        self.model.load_model(path)
        meta_path = path.replace('.json', '_meta.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                self.feature_columns = meta['feature_columns']
        self.is_trained = True
        print(f"  [XGB] Model loaded from {path}")


class ThresholdAnalyzer:
    """
    V4's real test: Compare baseline vs ML-filtered at multiple thresholds.
    Thresholds are chosen on train/val, then applied to test.
    """

    def analyze_thresholds(self, probabilities, labels, pnl_pcts, costs_per_trade=0):
        """
        Test multiple P(win) thresholds and find the optimal one.

        Args:
            probabilities: array of P(win) from XGBoost
            labels: array of actual outcomes (1=win, 0=loss)
            pnl_pcts: array of actual P&L percentages per trade
            costs_per_trade: fixed cost percentage per trade

        Returns:
            DataFrame comparing thresholds
        """
        thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
        results = []

        # Baseline (no filter)
        baseline_trades = len(labels)
        baseline_wins = labels.sum()
        baseline_pnl = pnl_pcts.sum()

        results.append({
            'threshold': 'BASELINE',
            'trades': baseline_trades,
            'wins': int(baseline_wins),
            'win_rate': baseline_wins / baseline_trades * 100 if baseline_trades > 0 else 0,
            'total_pnl_pct': baseline_pnl,
            'avg_pnl_pct': pnl_pcts.mean(),
        })

        for thresh in thresholds:
            mask = probabilities >= thresh
            filtered_labels = labels[mask]
            filtered_pnl = pnl_pcts[mask]

            trades = len(filtered_labels)
            if trades == 0:
                results.append({
                    'threshold': f'>={thresh:.2f}',
                    'trades': 0, 'wins': 0, 'win_rate': 0,
                    'total_pnl_pct': 0, 'avg_pnl_pct': 0,
                })
                continue

            wins = filtered_labels.sum()
            results.append({
                'threshold': f'>={thresh:.2f}',
                'trades': trades,
                'wins': int(wins),
                'win_rate': wins / trades * 100 if trades > 0 else 0,
                'total_pnl_pct': filtered_pnl.sum(),
                'avg_pnl_pct': filtered_pnl.mean(),
            })

        return pd.DataFrame(results)
