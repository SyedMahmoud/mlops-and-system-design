from src.load import load_data
from src.transform import Transformer, balance_dataset
from src.train import train_model
from src.store import store_model
from metadata import MODEL_NAME


def main():
    df = load_data(file_name="bank-full_train_test.csv")
    df = balance_dataset(df)
    df = Transformer().transform(df)
    lr_model = train_model(df=df, target_column="y")
    store_model(model=lr_model, model_name=MODEL_NAME)


# This allows to run this code only when the main.py file is executed
# It won't be executed when importing it
if __name__ == "__main__":
    main()
