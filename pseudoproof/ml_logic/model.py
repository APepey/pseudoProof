# basics
import pandas as pd
import numpy as np

# formatting
from typing import Tuple

# make stuff pretty
from colorama import Fore, Style

# sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import (
    cross_validate,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# tensorflow
from tensorflow import keras
from keras import Model, Sequential, layers, regularizers, optimizers
from keras.layers import Dense
from keras.callbacks import EarlyStopping


# All RNN-related functions, requiring specific parameters


def initialize_NNmodel(input_shape: tuple):
    """
    Initialize the Neural Network with random weights
    """
    rnn_model = Sequential()
    rnn_model.add(Dense(10, input_shape=input_shape, activation="relu"))
    rnn_model.add(Dense(20, activation="relu"))
    rnn_model.add(Dense(15, activation="relu"))
    rnn_model.add(Dense(8, activation="relu"))
    rnn_model.add(Dense(1, activation="sigmoid"))

    print("✅ RNN model initialized")

    return rnn_model


def compile_NNmodel(model: Model, learning_rate=0.005) -> Model:
    """
    Compile the Neural Network
    """
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    rnn_model = model.compile(
        loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )

    print("✅ RNN model compiled")

    return rnn_model


def fit_NNmodel(rnn_model, X_train, y_train, X_test, y_test):
    """
    Fit the RNN model to X_train and y_train.
    """

    fitted_rnn_model = rnn_model.fit(
        X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test)
    )

    print("✅ RNN model fitted")

    return fitted_rnn_model


def train_NNmodel(
    fitted_rnn_model: Model,
    X: np.ndarray,
    y: np.ndarray,
    batch_size=256,
    patience=2,
    validation_data=None,  # overrides validation_split
    validation_split=0.3,
) -> Tuple[Model, dict]:
    """
    Train the model on available data
    """
    es = EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1
    )

    history = fitted_rnn_model.fit(
        X,
        y,
        validation_data=validation_data,
        validation_split=validation_split,
        epochs=100,
        batch_size=batch_size,
        callbacks=[es],
        verbose=0,
    )
    print("✅ RNN model trained")

    trained_RNN_model = fitted_rnn_model

    return trained_RNN_model, history


def evaluate_RNNmodel(
    trained_RNN_model: Model, X: np.ndarray, y: np.ndarray, batch_size=64
) -> Tuple[Model, dict]:
    """
    Evaluate trained model performance on the dataset
    """

    print(Fore.BLUE + f"\nEvaluating model on {len(X)} rows..." + Style.RESET_ALL)

    if trained_RNN_model is None:
        print(f"\n❌ No model to evaluate")
        return None

    metrics = trained_RNN_model.evaluate(
        x=X, y=y, batch_size=batch_size, verbose=0, return_dict=True
    )

    acc = metrics["accuracy"]
    mae = metrics["mae"]

    print(f"✅ RNN model evaluated, MAE: {round(mae, 2)}, accuracy: {round(acc, 2)}")

    return metrics


# All 5 models that we use for basic prediction baselines

# define the differents models we can choose from
knn_model = KNeighborsClassifier(n_neighbors=2, p=1, weights='distance')
#Best Parameters KNN: {'n_neighbors': 2, 'p': 1, 'weights': 'distance'}
#Accuracy with Best Parameters KNN: 99.98%
nb_model = GaussianNB()
#For Naive Bayes it doesn't really make sense to make a grid search, as it only has one paramater ('priors')
#and is usually used in specific scenarios like data imbalance (to give more weight to the minority class), or
#weighting priors with external information that is not captured by training data (niche scenario). In any case,
# priors are calculated automatically during training so unless there's a good reason, we shouldn't use the param.
gbc_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05)
#Best Parameters for GBC: {'learning_rate': 0.05, 'n_estimators': 100}
#Accuracy with Best Parameters for GBC: 100%
svm_model = SVC(kernel="poly", C=10, gamma='scale')
#Best Parameters for SVM: {'C': 10, 'gamma': 'scale', 'kernel': 'poly'}
#Accuracy with Best Parameters for SVM: 85.88%
rf_model = RandomForestClassifier(max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=50)
#Best Parameters for Random Forest: {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 50}
#Accuracy with Best Parameters: 100%
#All these results came to be from the full comprehensive dataset on a train_test split.
#The models online won't be trained like this as it will


# create a dict to access models both as strings and variables
MODELS = {x§
    knn_model: "knn",
    nb_model: "naive_bayes",
    gbc_model: "gradient_boosting",
    rf_model: "random_forest",
    svm_model: "svm",
}


def fit(model_category, X_train, y_train):
    """
    Fit values according to one of the 5 model options
    """
    valid = {"gradient_boosting", "knn", "naive_bayes", "random_forest", "svm"}
    if model_category not in valid:
        raise ValueError("results: status must be one of %r." % valid)

    # fetching the model corresponding to the chosen parameter
    key_value_pairs = MODELS.items()
    for key, value in key_value_pairs:
        if value == model_category:
            model = key

    # fitting the model
    fitted_model = model.fit(X_train, y_train)  # will not work,

    print("✅ Model trained")

    return fitted_model


def evaluate(fitted_model, X, y, cv=5):
    """
    Evaluate the performace of the fitted model with a cross-validation.
    """
    score = cross_validate(fitted_model, X, y, cv)
    avg_test_score = score["test_score"].mean()

    print(f"✅ Model score over {cv} splits was {avg_test_score}")

    return avg_test_score
