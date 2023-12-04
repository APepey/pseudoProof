from pseudoproof.ml_logic.model_params import *
from pseudoproof.ml_logic.preproc import *
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import pickle
import pandas as pd
import tqdm
import os


def best_params(model_category, X_train, y_train):
    if model_category == 'naive_bayes':
        params = nb_param_grid
        model = GaussianNB()
    elif model_category == 'svm':
        params = svm_param_grid
        model = SVC()
    elif model_category == 'gradient_boosting':
        params = gb_param_grid
        model = GradientBoostingClassifier()
    elif model_category == 'knn':
        params = knn_param_grid
        model = KNeighborsClassifier()
    elif model_category == 'random_forest':
        params = rf_param_grid
        model = RandomForestClassifier()


    grid = RandomizedSearchCV(estimator=model,
                         param_distributions=params, cv=5,
                         scoring='accuracy', n_iter=600, random_state=6, n_jobs=-1)
    grid.fit(X_train, y_train)
    best_score = grid.best_score_
    # Save model as pickle file
    model_name = model_category + f'_{round(best_score,2)}.pkl'
    model_path = os.path.join('grid_search_models', model_name)


    with open(model_path, 'wb') as file:
        pickle.dump(grid.best_estimator_, file)
    print(f'Saved {model_category} as pickle file')



    return {'best_params':grid.best_params_,
            'best_score':grid.best_score_}


def record_model_params():
    df = pd.read_csv("pseudoproof/ml_logic/all_data.csv")

    X = df.drop(columns=['y'])
    y = df['y']

    results = {}
    for model in tqdm.tqdm(['naive_bayes', 'svm', 'gradient_boosting', 'knn', 'random_forest']) :
         results[model] = best_params(model, X, y)
         print(f'model: {results[model] }' )


if __name__ == '__main__':
    record_model_params()
