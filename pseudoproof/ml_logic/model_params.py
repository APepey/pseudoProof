from scipy.stats import randint, loguniform

nb_param_grid = {'var_smoothing':loguniform(1e-20, 20)}

#Specifying the possible Naive Bayes param values the RandomGridSearch will test
svm_param_grid = {'C': loguniform(90.0, 500.0),
                  'kernel':['linear', 'poly', 'rbf'],
                  'degree': randint(1,5),
                  'coef0':loguniform(0.001,0.1),
                   'tol':loguniform(1e-3,1.0), }


#Specifyung the possible GB param values the RandomGridSearch will test
gb_param_grid = {'n_estimators': randint(10,500),
              'learning_rate':loguniform(0.01,1)}


#Specifyung the possible KNN param values the RandomGridSearch will test
knn_param_grid = {'n_neighbors': randint(1, 25),
                  'leaf_size': randint(1, 30),
                  'weights': ['uniform', 'distance'],
                  'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}


#Specifying the possible Random Forest param values the RandomGridSearch will test
rf_param_grid = {'n_estimators': randint(80,400),
                 'random_state': randint(1,10)}
