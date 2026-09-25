"""Core module for the Data Doctor library."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


class Doctor:
    """
    The core class for diagnosing, treating, and analyzing tabular data.

    Example
    -------
    >>> import pandas as pd
    >>> from datadoctor import Doctor
    >>> df = pd.read_csv("messy.csv")
    >>> doc = Doctor(df)
    >>> doc.diagnose()
    >>> clean = doc.treat()
    """

    def __init__(self, df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        self.df: pd.DataFrame = df.copy()
        self._diagnosis: dict = {}

    # ------------------------------------------------------------------
    # Diagnosis
    # ------------------------------------------------------------------
    def diagnose(self) -> dict:
        """
        Performs a full diagnostic check on the DataFrame.

        Returns a report containing row/column counts, missing values,
        data types, duplicate rows, and IQR-based outlier counts.
        """
        report = {
            "num_rows": int(len(self.df)),
            "num_columns": int(len(self.df.columns)),
            "missing_values": {
                col: int(count) for col, count in self.df.isnull().sum().items()
            },
            "data_types": {
                col: str(dtype) for col, dtype in self.df.dtypes.items()
            },
            "duplicate_rows": int(self.df.duplicated().sum()),
            "outliers_iqr": self._count_outliers_iqr(),
        }
        self._diagnosis = report
        print("Diagnosis complete.")
        return report

    def _count_outliers_iqr(self) -> dict:
        """Counts IQR-based outliers for every numeric column."""
        counts: dict = {}
        for col in self.df.select_dtypes(include="number").columns:
            series = self.df[col].dropna()
            if series.empty:
                counts[col] = 0
                continue
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                counts[col] = 0
                continue
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            counts[col] = int(((series < lower) | (series > upper)).sum())
        return counts

    # ------------------------------------------------------------------
    # Treatment
    # ------------------------------------------------------------------
    def convert_types(self) -> pd.DataFrame:
        """
        Infers and converts column types.

        - Numeric-looking text columns become numeric.
        - Date-looking text columns become datetime.
        A column is only converted if at least 90% of its values parse
        successfully, which prevents destroying genuinely textual columns.
        """
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                continue
            if pd.api.types.is_datetime64_any_dtype(self.df[col]):
                continue

            series = self.df[col]
            threshold = 0.9 * len(series)

            # Try numeric conversion first
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.notna().sum() >= threshold:
                self.df[col] = numeric
                continue

            # Then try datetime conversion
            try:
                with pd.option_context("mode.chained_assignment", None):
                    datetimes = pd.to_datetime(series, errors="coerce")
                if datetimes.notna().sum() >= threshold:
                    self.df[col] = datetimes
            except Exception:
                # Column stays as-is if it cannot be parsed
                pass

        print("Type conversion complete.")
        return self.df

    def detect_outliers(self, method: str = "iqr") -> dict:
        """
        Detects outliers in numeric columns.

        Args:
            method: "iqr" (1.5 * IQR rule) or "zscore" (|z| > 3).

        Returns:
            A dict mapping column names to lists of outlier row indices.
        """
        if method not in ("iqr", "zscore"):
            raise ValueError("Method must be 'iqr' or 'zscore'.")

        outliers: dict = {}
        for col in self.df.select_dtypes(include="number").columns:
            series = self.df[col]

            if method == "iqr":
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                if iqr == 0:
                    mask = pd.Series(False, index=series.index)
                else:
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    mask = (series < lower) | (series > upper)
            else:
                std = series.std()
                if std == 0 or pd.isna(std):
                    mask = pd.Series(False, index=series.index)
                else:
                    z = (series - series.mean()).abs() / std
                    mask = z > 3

            outliers[col] = self.df.index[mask.fillna(False)].tolist()

        print(f"Outlier detection complete using {method}.")
        return outliers

    def treat_outliers(self, method: str = "iqr") -> pd.DataFrame:
        """
        Caps outliers to the IQR bounds (winsorization).

        Values below the lower bound are raised to the lower bound, and
        values above the upper bound are lowered to the upper bound. This
        preserves row count while removing the distorting effect of extremes.
        """
        if method != "iqr":
            raise ValueError("Only 'iqr' winsorization is currently supported.")

        for col in self.df.select_dtypes(include="number").columns:
            series = self.df[col]
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            self.df[col] = series.clip(lower=lower, upper=upper)

        print("Outlier treatment complete (values capped to IQR bounds).")
        return self.df

    def treat(self, strategy: str = "auto") -> pd.DataFrame:
        """
        Cleans the DataFrame.

        Args:
            strategy: "auto" performs type conversion, missing value
                imputation, and deduplication. "aggressive" additionally
                winsorizes outliers.

        Returns:
            The cleaned DataFrame.
        """
        # Step 1: Type conversion
        self.convert_types()

        # Step 2: Re-diagnose after conversion, since types may have changed
        self.diagnose()

        # Step 3: Missing value imputation
        for col, missing_count in self._diagnosis["missing_values"].items():
            if missing_count == 0:
                continue

            if self.df[col].isnull().all():
                # Nothing to impute from; drop the empty column
                self.df = self.df.drop(columns=[col])
                continue

            if pd.api.types.is_numeric_dtype(self.df[col]):
                self.df[col] = self.df[col].fillna(self.df[col].median())
            elif pd.api.types.is_datetime64_any_dtype(self.df[col]):
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
            else:
                modes = self.df[col].mode()
                if not modes.empty:
                    self.df[col] = self.df[col].fillna(modes[0])

        # Step 4: Deduplication
        self.df = self.df.drop_duplicates().reset_index(drop=True)

        # Step 5: Optional outlier treatment
        if strategy == "aggressive":
            self.treat_outliers(method="iqr")

        print("Treatment complete. Data cleaned.")
        return self.df

    # ------------------------------------------------------------------
    # Modeling
    # ------------------------------------------------------------------
    def export_model(
        self,
        target_column: str,
        model_path: str = "model.joblib",
        use_smote: bool = False,
    ):
        """
        Trains a baseline RandomForestClassifier and saves it to disk.

        Args:
            target_column: Name of the column to predict.
            model_path: Where to write the trained model.
            use_smote: If True, applies SMOTE to the TRAINING SET ONLY.
                Requires the optional `imbalanced-learn` dependency.

        Returns:
            The trained model.
        """
        if target_column not in self.df.columns:
            raise ValueError(f"Column '{target_column}' not found in DataFrame.")

        X = self.df.drop(columns=[target_column])
        y = self.df[target_column]

        # One-hot encode categorical features
        X = pd.get_dummies(X, drop_first=True)
        # Ensure every feature is numeric
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

        # Split BEFORE any resampling to avoid data leakage
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        if use_smote:
            from .imbalance import apply_smote

            X_train, y_train = apply_smote(X_train, y_train)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)
        print(f"Model accuracy on held-out test set: {accuracy:.4f}")

        joblib.dump(model, model_path)
        print(f"Model trained and saved to {model_path}")
        return model

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def summary(self) -> str:
        """Returns a short human-readable summary of the current DataFrame."""
        lines = [
            f"Rows: {len(self.df)}",
            f"Columns: {len(self.df.columns)}",
            f"Missing values: {int(self.df.isnull().sum().sum())}",
            f"Duplicate rows: {int(self.df.duplicated().sum())}",
        ]
        return "\n".join(lines)