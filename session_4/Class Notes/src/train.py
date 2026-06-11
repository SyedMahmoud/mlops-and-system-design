import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from metadata import MODEL_PARAMS


def train_model(df: pd.DataFrame, target_column: str) -> LogisticRegression:
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LogisticRegression(**MODEL_PARAMS)  # Increase max_iter if needed
    model.fit(X_train, y_train)

    return model
