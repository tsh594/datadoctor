"""Optional class-imbalance utilities.

This module is only imported when `export_model(use_smote=True)` is called,
so the base library install stays lightweight.
"""

from __future__ import annotations

import pandas as pd


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
):
    """
    Applies SMOTE to the training data ONLY.

    Never apply SMOTE to the test set — doing so causes data leakage and
    invalidates your evaluation metrics.

    Args:
        X_train: Training features (numeric or one-hot encoded).
        y_train: Training target labels.
        random_state: Seed for reproducibility.

    Returns:
        A tuple of (resampled X_train, resampled y_train).
    """
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError as exc:
        raise ImportError(
            "SMOTE requires the optional dependency 'imbalanced-learn'. "
            "Install it with: pip install 'datadoctor[imbalance]'"
        ) from exc

    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print(
        f"SMOTE applied. Original shape: {X_train.shape}, "
        f"Resampled shape: {X_resampled.shape}"
    )
    return X_resampled, y_resampled