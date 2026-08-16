from collections.abc import Sequence

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted


class NativeCategoryTransformer(BaseEstimator, TransformerMixin):
    """Prepare categorical columns for LightGBM native categoricals."""

    def __init__(
        self,
        columns: Sequence[str],
        missing_value: str = "__MISSING__",
    ):
        self.columns = columns
        self.missing_value = missing_value

    def fit(self, X: pd.DataFrame, y=None):
        missing_columns = set(self.columns) - set(X.columns)

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {sorted(missing_columns)}"
            )

        # Learn the categories present in the training data.
        self.categories_ = {}

        for column in self.columns:
            values = X[column].fillna(self.missing_value).astype(str)
            self.categories_[column] = list(pd.unique(values))

        # Remember the complete training schema and column order.
        self.feature_names_in_ = list(X.columns)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "categories_")

        missing_columns = set(self.feature_names_in_) - set(X.columns)

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {sorted(missing_columns)}"
            )

        # Select and order inputs exactly as they appeared during training.
        result = X.loc[:, self.feature_names_in_].copy()

        for column in self.columns:
            values = result[column].fillna(self.missing_value).astype(str)

            # Reuse the categories learned during fit().
            result[column] = pd.Categorical(
                values,
                categories=self.categories_[column],
            )

        return result


def make_preprocessing_pipeline(
    categorical_columns: Sequence[str],
) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "native_categories",
                NativeCategoryTransformer(
                    columns=categorical_columns
                ),
            ),
        ]
    )