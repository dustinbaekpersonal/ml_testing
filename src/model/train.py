import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.dataloader.preprocess import col_transform, label_aggregate, make_dataset


def trainer(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    clf = RandomForestClassifier()
    _, preprocessor = col_transform(X_train)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifer", clf),
        ]
    )
    y_train = label_aggregate(y_train)
    pipeline.fit(X_train, y_train)
    return pipeline


if __name__ == "__main__":
    (X_train, X_test, y_train, y_test) = make_dataset()
    pipeline = trainer(X_train, y_train)
    y_test = label_aggregate(y_test)
    print(f"model score: {pipeline.score(X_test, y_test):.3f}")
