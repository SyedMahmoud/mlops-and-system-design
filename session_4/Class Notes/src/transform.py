import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from metadata import (
    COLUMNS_TO_DROP,
    BINARY_FEATURES,
    ONE_HOT_ENCODE_COLUMNS,
)


class Transformer:
    def __init__(self):
        self.drop_columns = COLUMNS_TO_DROP
        self.binary_variable_columns = BINARY_FEATURES
        self.one_hot_encoding_columns = ONE_HOT_ENCODE_COLUMNS

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop(self.drop_columns, axis=1)
        df = self._map_binary_column_to_int(df)
        df = self._map_month_to_int(df)
        df = self._one_hot_encoding(df)

        return df

    def _map_binary_column_to_int(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.binary_variable_columns:
            df[col] = df[col].map({"yes": 1, "no": 0})
        return df

    def _map_month_to_int(self, df: pd.DataFrame) -> pd.DataFrame:
        month_mapping = {
            "jan": 1,
            "feb": 2,
            "mar": 3,
            "apr": 4,
            "may": 5,
            "jun": 6,
            "jul": 7,
            "aug": 8,
            "sep": 9,
            "oct": 10,
            "nov": 11,
            "dec": 12,
        }
        df["month"] = df["month"].map(month_mapping)

        return df

    def _one_hot_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        encoder = OneHotEncoder(drop="first", sparse_output=False).set_output(
            transform="pandas"
        )
        encoder.fit(df[ONE_HOT_ENCODE_COLUMNS])
        encoded_df = encoder.transform(df[ONE_HOT_ENCODE_COLUMNS])
        df = df.drop(columns=ONE_HOT_ENCODE_COLUMNS)
        df = pd.concat([df, encoded_df], axis=1)

        return df


def balance_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # Separate the classes
    df_y0 = df[df["y"] == "no"].copy()
    df_y1 = df[df["y"] == "yes"].copy()

    # Find the smaller class size
    min_size = len(df_y1)

    # Randomly sample from each class
    df_y0_balanced = df_y0.sample(n=min_size, random_state=42)

    # Concatenate back together
    df_balanced = pd.concat([df_y0_balanced, df_y1])

    # Shuffle the dataset
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    return df_balanced
