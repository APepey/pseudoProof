#Importing libraries
import numpy as np
import pandas as pd

#sklearn
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
#scipy
from scipy.stats import randint, loguniform #Importing libraries for RandomGridSeaerch

#functions
from ml_logic.model import *
from ml_logic.preproc import *
from ml_logic.gridsearch import best_params




#
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
