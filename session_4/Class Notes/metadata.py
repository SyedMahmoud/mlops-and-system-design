MODELS_FOLDER = "session_4/models"
DATASETS_FOLDER = "session_4/datasets"
MODEL_NAME = "logistic-regression-model"

COLUMNS_TO_DROP = []
BINARY_FEATURES = [
    "housing",
    "loan",
    "default",
]
ONE_HOT_ENCODE_COLUMNS = [
    "marital",
    "job",
    "education",
    "poutcome",
    "contact",
]
MODEL_PARAMS = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "multi_class": "auto",
    "random_state": 8888,
}
