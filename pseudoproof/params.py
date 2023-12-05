import os

LOCAL_MODEL_PATH = os.path.expanduser("~/.lewagon/saved_models")
os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)

# create a dict to access models both as strings and variables
# MODELS = {
#     knn_model: "knn",
#     nb_model: "naive_bayes",
#     gbc_model: "gradient_boosting",
#     rf_model: "random_forest",
#     svm_model: "svm",
# }

# LOCAL_REGISTRY_PATH = os.path.join(
#     os.path.expanduser("~"), ".lewagon", "mlops", "training_outputs"
# )
# MODEL_TARGET = os.environ.get("MODEL_TARGET")
# BUCKET_NAME = os.environ.get("BUCKET_NAME")
