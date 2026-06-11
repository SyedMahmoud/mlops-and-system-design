from src.transform import Transformer, balance_dataset
import pandas as pd


def test_map_binary_column_to_int():
    transformer = Transformer()
    df = pd.DataFrame(
        {
            "housing": ["yes", "no", "yes", "no"],
            "loan": ["no", "yes", "yes", "no"],
            "default": ["no", "yes", "yes", "no"],
        }
    )

    expected_df = pd.DataFrame(
        {"housing": [1, 0, 1, 0], "loan": [0, 1, 1, 0], "default": [0, 1, 1, 0]}
    )

    transformed_df = transformer._map_binary_column_to_int(df)

    # Test the result against the expected DataFrame
    pd.testing.assert_frame_equal(transformed_df, expected_df)


def test_map_month_to_int():
    transformer = Transformer()
    df = pd.DataFrame({"month": ["jan", "feb", "mar", "apr"]})

    expected_df = pd.DataFrame({"month": [1, 2, 3, 4]})

    transformed_df = transformer._map_month_to_int(df)

    # Test the result against the expected DataFrame
    pd.testing.assert_frame_equal(transformed_df, expected_df)


def create_df_balance():
    return pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 45, 50],
            "job": [
                "admin",
                "technician",
                "admin",
                "technician",
                "admin",
                "technician",
            ],
            "y": ["yes", "no", "yes", "no", "yes", "no"],
        }
    )


def test_balance_dataset():

    df_balance = create_df_balance()
    # Define the expected balanced output
    expected_df = pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 45, 50],
            "job": [
                "admin",
                "technician",
                "admin",
                "technician",
                "admin",
                "technician",
            ],
            "y": ["yes", "no", "yes", "no", "yes", "no"],
        }
    )
    expected_df = expected_df.sort_values(["age", "job"]).reset_index(drop=True)

    # Balance the dataset
    balanced_df = balance_dataset(df_balance)
    balanced_df = balanced_df.sort_values(["age", "job"]).reset_index(drop=True)

    # Test the result against the expected DataFrame (order may vary)
    pd.testing.assert_frame_equal(
        balanced_df.sort_index(axis=1), expected_df.sort_index(axis=1)
    )


def test_balance_with_unequal_classes():
    df_unequal = pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 45],
            "job": ["admin", "technician", "admin", "technician", "admin"],
            "y": ["no", "no", "yes", "no", "yes"],
        }
    )

    # Balance the dataset
    balanced_df = balance_dataset(df_unequal)

    # Check if classes are balanced
    class_counts = balanced_df["y"].value_counts()
    assert class_counts["yes"] == class_counts["no"]
