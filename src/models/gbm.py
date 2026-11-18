"""
Gradient-boosted regressor used for the spread (margin), totals and props
models -- same setup as the CFB game and totals models: fixed seed, missing
inputs filled with training medians, residual spread measured on a HOLDOUT
(not training rows; the CFB model's 9/2026 overconfidence bug came from
training residuals).
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split


class GBMModel:
    def __init__(self, features, target, n_estimators=300, max_depth=3, learning_rate=0.04):
        self.features_wanted = list(features)
        self.target = target
        self.model = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                                               subsample=0.8, min_samples_leaf=20, random_state=42)
        self.feature_columns, self.medians = [], {}
        self.residual_std, self.n_train, self.holdout_mae = None, 0, None

    def _X(self, df):
        X = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
                          for c in self.feature_columns}, index=df.index)
        return X.fillna(self.medians)

    def fit(self, df: pd.DataFrame, verbose=True):
        d = df[pd.to_numeric(df[self.target], errors="coerce").notna()]
        self.feature_columns = [c for c in self.features_wanted if c in d.columns and pd.to_numeric(d[c], errors="coerce").notna().any()]
        if not self.feature_columns:
            raise ValueError("no usable inputs")
        self.medians = {c: float(pd.to_numeric(d[c], errors="coerce").median()) for c in self.feature_columns}
        X, y = self._X(d), d[self.target].astype(float)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(Xtr, ytr)
        res = yte - self.model.predict(Xte)
        self.residual_std = float(np.std(res))
        self.holdout_mae = float(np.mean(np.abs(res)))
        self.model.fit(X, y)          # final fit on everything
        self.n_train = int(len(d))
        if verbose:
            print(f"[{self.target}] {self.n_train} rows, holdout MAE {self.holdout_mae:.2f}, residual sd {self.residual_std:.2f}")
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self._X(df))

    def importances(self, top=10):
        pairs = sorted(zip(self.feature_columns, self.model.feature_importances_), key=lambda kv: -kv[1])
        return {k: round(float(v), 4) for k, v in pairs[:top]}

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
