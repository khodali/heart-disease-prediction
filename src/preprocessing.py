from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class ZeroToNaNTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns if columns is not None else ["RestingBP", "Cholesterol"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            if col in X.columns:
                # تبدیل 0 به NaN برای کارکرد درست Imputer در مراحل بعدی خط لوله
                X[col] = X[col].replace(0, np.nan)
        return X
