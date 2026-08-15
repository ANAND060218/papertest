"""
V4 -- Walk-Forward Validation
Proves the edge survives across different market conditions.

Walk-forward splits:
  Train 2016-2020 -> Test 2021
  Train 2016-2021 -> Test 2022
  Train 2016-2022 -> Test 2023
  Train 2016-2023 -> Test 2024
  Train 2016-2024 -> Test 2025-26

If performance degrades severely in any window, the edge is likely spurious.
"""
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class WalkForwardValidator:
    """
    Walk-forward validation for time-series trading data.
    Never uses future data for training.
    """

    def generate_splits(self, df, date_col='timestamp',
                        train_start_year=2016, first_test_year=2021, last_year=2026):
        """
        Generate expanding window train/test splits.

        Yields: (train_mask, test_mask, split_name)
        """
        dates = pd.to_datetime(df[date_col])

        for test_year in range(first_test_year, last_year + 1):
            train_mask = dates.dt.year < test_year
            test_mask = dates.dt.year == test_year

            if train_mask.sum() == 0 or test_mask.sum() == 0:
                continue

            train_years = f"{train_start_year}-{test_year - 1}"
            split_name = f"Train({train_years}) -> Test({test_year})"

            yield train_mask, test_mask, split_name

    def run_walk_forward(self, trades_df, feature_cols, label_col='label',
                         model_class=None, date_col='timestamp'):
        """
        Run walk-forward validation on labeled trades.

        Args:
            trades_df: DataFrame with features + label + timestamp
            feature_cols: list of feature column names
            label_col: column with 0/1 labels
            model_class: class with train() and predict_proba() methods
            date_col: timestamp column

        Returns:
            List of per-split results
        """
        results = []

        for train_mask, test_mask, split_name in self.generate_splits(trades_df, date_col):
            train_df = trades_df[train_mask]
            test_df = trades_df[test_mask]

            X_train = train_df[feature_cols].fillna(0)
            y_train = train_df[label_col]
            X_test = test_df[feature_cols].fillna(0)
            y_test = test_df[label_col]

            print(f"\n{'=' * 70}")
            print(f"  {split_name}")
            print(f"  Train: {len(train_df)} trades | Test: {len(test_df)} trades")

            if len(train_df) < 10 or len(test_df) < 3:
                print(f"  [SKIP] Insufficient data for this split")
                continue

            # Train model
            model = model_class()
            model.train(X_train, y_train)

            # Predict on test
            test_proba = model.predict_proba(X_test)
            test_pred = (test_proba >= 0.5).astype(int)

            # Metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score

            accuracy = accuracy_score(y_test, test_pred)
            precision = precision_score(y_test, test_pred, zero_division=0)
            recall = recall_score(y_test, test_pred, zero_division=0)

            # P&L comparison: baseline vs filtered
            baseline_pnl = test_df['pnl_pct'].sum() if 'pnl_pct' in test_df.columns else 0
            filtered_mask = test_proba >= 0.55
            filtered_pnl = test_df[filtered_mask]['pnl_pct'].sum() if 'pnl_pct' in test_df.columns else 0

            split_result = {
                'split': split_name,
                'train_size': len(train_df),
                'test_size': len(test_df),
                'accuracy': round(accuracy, 3),
                'precision': round(precision, 3),
                'recall': round(recall, 3),
                'baseline_pnl_pct': round(baseline_pnl, 4),
                'filtered_pnl_pct': round(filtered_pnl, 4),
                'improvement': round(filtered_pnl - baseline_pnl, 4),
            }
            results.append(split_result)

            print(f"  Accuracy:  {accuracy:.3f}")
            print(f"  Precision: {precision:.3f}")
            print(f"  Recall:    {recall:.3f}")
            print(f"  Baseline P&L:  {baseline_pnl:+.4f}%")
            print(f"  Filtered P&L:  {filtered_pnl:+.4f}%")
            print(f"  ML Improvement: {filtered_pnl - baseline_pnl:+.4f}%")

        return results

    def summarize(self, results):
        """Print walk-forward summary."""
        if not results:
            print("\n[WF] No walk-forward results to summarize.")
            return

        df = pd.DataFrame(results)
        print(f"\n{'=' * 90}")
        print("WALK-FORWARD VALIDATION SUMMARY")
        print(f"{'=' * 90}")
        print(df.to_string(index=False))

        # Overall
        total_improvement = df['improvement'].sum()
        avg_accuracy = df['accuracy'].mean()
        consistent = (df['improvement'] > 0).sum()

        print(f"\nAvg Accuracy: {avg_accuracy:.3f}")
        print(f"Total ML Improvement: {total_improvement:+.4f}%")
        print(f"Consistent improvements: {consistent}/{len(df)} splits")

        if consistent > len(df) / 2 and total_improvement > 0:
            print("\n>>> ML ADDS VALUE -- Proceed to V5 (EV)")
        else:
            print("\n>>> ML does NOT consistently improve results")
            print("    Consider: different features, different model, or stay rule-based")

        return df
