"""This python file is to data load and data preprocessing"""
import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.configs.config import load_config


def load_data() -> pd.DataFrame:
    """Loading csv files as pandas DataFrame"""
    config = load_config()
    df_red = pd.read_csv(os.path.join(config["data_path"], config["data_name"][0]), sep=";")
    df_white = pd.read_csv(os.path.join(config["data_path"], config["data_name"][1]), sep=";")
    return df_red, df_white


def load_df() -> pd.DataFrame:
    """
    Merge red csv and white csv files by creating "type" column
    type column (categorical): red or white
    """
    df_red, df_white = load_data()
    df_red.insert(0, "type", "red")
    df_white.insert(0, "type", "white")
    df = pd.concat([df_red, df_white])
    return df


def make_dataset() -> pd.DataFrame:
    """
    Split raw dataframe into train, validation, test sets
    80/10/10 split
    """
    df = load_df()
    y = df["quality"].astype(int)
    X = df.drop("quality", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    return X_train, X_test, y_train, y_test


class CustomFeatureSubtract(BaseEstimator, TransformerMixin):
    """
    This is class for custom sklearn Transformer that creates new feature
    by subtracting total sulfur dioxide and free sulfur dioxide
    """

    def __init__(self, feature_one: str, feature_two: str):
        self.feature_one = feature_one
        self.feature_two = feature_two

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        new_feature_name = self.feature_two + " minus " + self.feature_one
        X[new_feature_name] = X[self.feature_two] - X[self.feature_one]
        return X


def col_transform(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, ColumnTransformer]:
    """
    One hot encodes categorical column
    Standardize numerical column
    Args:
        df_raw (pd.DataFrame): input raw dataframe
    Returns:
        df_trans (np.ndarray): transformed pandas dataframe
        ct (ColumnTransformer): preprocessor to be part of pipeline in train.py
    """

    cat_features = [
        col for col in df_raw.columns if df_raw.dtypes[col] in ["object", "string", "bool", "category"]
    ]
    num_features = [
        col for col in df_raw.columns if df_raw.dtypes[col] not in ["object", "string", "bool", "category"]
    ]
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(missing_values=np.nan, strategy="mean")),
            (
                "feature_subtract",
                CustomFeatureSubtract(feature_one="free sulfur dioxide", feature_two="total sulfur dioxide"),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            # ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )

    ct = ColumnTransformer(
        remainder="passthrough",
        transformers=[
            ("numeric", numeric_transformer, num_features),
            ("categorical", categorical_transformer, cat_features),
        ],
        n_jobs=-1,
    )
    df_trans = ct.fit_transform(df_raw)
    return df_trans, ct


def label_aggregate(label: pd.Series) -> pd.Series:
    """
    Aggregate label(quality) to 0(Low), 1(Medium), 2(High)
    Args:
        label (pd.Series): Series with column "quality", i.e. df["quality"]
    Returns:
        label (pd.Series): Seires with values converted to 0, 1, 2
    """

    def quality_mapping(x: int) -> int:
        return 0 if x < 5 else 1 if x >= 5 and x < 7 else 2

    if label[(label <= 0) | (label >= 10)].any():
        raise ValueError("Quality column is integer from 1 to 9")

    label = pd.Series(list(map(quality_mapping, label)), index=label.index)
    return label


if __name__ == "__main__":
    (X_train, X_test, y_train, y_test) = make_dataset()
    print(X_train.head(1))
    custom = CustomFeatureSubtract(feature_one="free sulfur dioxide", feature_two="total sulfur dioxide")
    asdf = custom.fit(X_train)
    print(asdf)

    # X_train, _ = col_transform(X_train)
    # print(X_train.head(1))
    # y_train = label_aggregate(y_train)
    # print(y_train)
    # asdf = pd.DataFrame({"col": [np.nan, 1, 3]})
    # print(asdf.info())
    # test_data = pd.DataFrame({"asdf": ["red", "white", "red", "blue"]}).astype("category")
    # asdf = pd.Series([0, 10], name="quality")
    # asdf, _ = label_aggregate(asdf)
    # print(asdf)
