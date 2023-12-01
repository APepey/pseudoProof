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
from ml_logic.params import *
from ml_logic.preproc import *
from ml_logic.gridsearch import best_params



# 1. DATA PREPROCESSING


#Load, clean, and scale data
df = pd.read_csv('all_data.csv')
df = clean_data(df)
df = scale_data(df)


#Define X and y
X = df.drop(columns=['y'])
y = df['y']


#Split data
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=6,
                                                    stratify = y)



#2 . MODEL TUNING

'''Now that we have precessed our data, we can start building our models.
We will use the following models:
- Random Forest
- Gradient Boosting
- KNN
- Naive Bayes
- SVM
- Neural Network
'''


#2.1 Random Forest

rf_model = RandomForestClassifier(best_params**("random_forest", X_train, y_train))







# 3. MODEL EVALUATION
'''
Given the nature of our data, we chose to use accuracy as our metric,
given that we're working on a classification task with balanced classes.

'''
