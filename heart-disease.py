# for data ready
import pandas as pd
import matplotlib.pyplot as plt

# for machine learning model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score

# for improve our model
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# for save our model
import pickle
import os

# setup random seed
import numpy as np

np.random.seed(42)

# making a function for train the model
def train_model(x_train, y_train):
    '''
    This function can be train our machine learning model.
    :return: trained model
    '''
    # make and fit the model
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    return model

# making a function for evaluate the function
def evaluate_model(y_true, y_preds):
    '''
    this function can compares the true labels and predicted labels.
    :param y_true: true lebels
    :param y_preds: predicted labels
    :return: evaluate matrix
    '''
    acc = accuracy_score(y_true, y_preds)
    precision = precision_score(y_true, y_preds, zero_division=1, average="weighted")
    recall = recall_score(y_true, y_preds, zero_division=1, average="weighted")
    f1 = f1_score(y_true, y_preds, zero_division=0, average="weighted")

    evaluate_matrix = {"Accuracy": acc,
                       "Precision": precision,
                       "Recall": recall,
                       "F1": f1
                      }

    print(f"Accuracy : {acc*100:.2f}%")
    print(f"Precision : {precision:.2f}")
    print(f"Recall : {recall:.2f}")
    print(f"F1_score : {f1:.2f}")

    return evaluate_matrix

# function for RandomizedSearchcv
def improved_Randomizedsearvhcv(x_validation, y_validation, x_test, y_test, grid):
    '''
    This function can improve our model with RandomizedSearchCV and evaluate our improved model.
    :return: improved model, best params and evaluation matrix
    '''
    print("\nModel start Improving through RandomizedSearchCV")
    print("-----------------------------------------------------------------------------------------------------------")
    # make the model
    model = RandomForestClassifier(n_jobs=-1)

    # setup randomizedserchCV
    rs_model = RandomizedSearchCV(estimator=model,
                                  param_distributions=grid,
                                  n_iter=10,
                                  verbose=2,
                                  cv=5
                                  )
    # fit the randomizedsearchcv version of clf
    rs_model.fit(x_validation, y_validation)

    # print the best params
    print("\nBest params from RandomizedSearch", rs_model.best_params_)

    # make predictions and evaluate our improved model
    rs_preds = rs_model.predict(x_test)
    print("\nRandomizedSearch model evaluation")
    evaluate_matrix = evaluate_model(y_test, rs_preds)

    # return the model and evaluate matrix
    return rs_model, evaluate_matrix

# function for gridSearchcv
def improve_gridsearchcv(x_validation, y_validation, x_test, y_test, grid):
    '''
    This function can improve our model with GridSearchCV and evaluate our improved model.
    :return: improved model, best params and evaluate matrix
    '''
    print("\nModel Start Improving through GridSearchCV")
    print("-----------------------------------------------------------------------------------------------------------")
    # make the model
    model = RandomForestClassifier(n_jobs=-1)

    # setup the gridsearchcv
    gs_model = GridSearchCV(estimator=model,
                           param_grid=grid,
                            verbose=2,
                            cv=5
                           )

    # fit the gridsearchcv version of clf
    gs_model.fit(x_validation, y_validation)

    # print the best params
    print("\nBest params from GridSearchCV:", gs_model.best_params_)

    # make predictions and evaluate our improved model
    gs_preds = gs_model.predict(x_test)
    print("\nGridSearch model evaluation")
    evaluate_matrix = evaluate_model(y_test, gs_preds)

    # return the model and evaluate matrix
    return gs_model, evaluate_matrix

# make a function for visualize the comparison of the evaluations of all models
def comp_matrix(baseline_matrix, rs_matrix, gs_matrix):
    '''
    This function can visualize comparison of the evaluations of all models for better understanding.
    :return: return the plot graph
    '''
    # make a dataframe of all the evaluation matrix
    model_comparison = pd.DataFrame({
        "Baseline": baseline_matrix,
        "RandomizedSearchCV": rs_matrix,
        "GridSearchCV": gs_matrix
    })

    # show the model comparison with dataframe
    print("\nAll Three Model comparison ")
    print(model_comparison)

    # plot and visualize our evaluations
    model_comparison.plot(kind="bar", figsize=(8, 6))
    plt.show()

# create a function for saving our improved model
def save_model(rs_model, gs_model, rs_matrix, gs_matrix):
    '''
    This function can compares the accuracy of all improved models and save the model which has best accuracy.
    '''
    print("\n")
    Model_name = "heart-disease-model.pkl"
    if os.access(Model_name, os.F_OK):
        print("This Model is already exist.")
    else:
        if rs_matrix["Accuracy"] > gs_matrix["Accuracy"]:
            print("RandomizedSearch Model has best accuracy.")
            pickle.dump(rs_model, open("heart-disease-model.pkl", "wb"))
            print("RandomizedSearch Model is saved successfully!!!!!!")
        elif rs_matrix["Accuracy"] == gs_matrix["Accuracy"]:
            print("GridSearch and RandomizedSearch model has similar accuracy then we will save GridSearch Model.")
            '''
            we can save GridSearch model over RandomizedSearch model because we know that the GridSearchCV make more sense
            for finding patterns with all the sets which are made by the grid.
            '''
            pickle.dump(gs_model, open("heart-disease-model.pkl", "wb"))
            print("GridSearch Model is saved Successfully!!!!!!!!")
        else:
            print("GridSearch Model has best accuracy.")
            pickle.dump(gs_model, open("heart-disease-model.pkl", "wb"))
            print("GridSearch Model is saved Successfully!!!!!!")

if __name__ == '__main__':
    # import the data
    data = pd.read_csv("../Data/heart-disease.csv")

    # shuffled the data
    data_shuffled = data.sample(frac=1)

    # make the data
    x = data.drop("target", axis=1)
    y = data["target"]

    # split the data
    train_split = round(0.7*len(data_shuffled))
    valid_split = round(train_split+0.15*len(data_shuffled))

    x_train, y_train = x[:train_split], y[:train_split]
    x_val, y_val = x[train_split:valid_split], y[train_split:valid_split]
    x_test, y_test = x[valid_split:], y[valid_split:]

    # train a machine learning model
    model = train_model(x_train, y_train)
    print(f"Model Accuracy : {model.score(x_test, y_test)*100:.2f}%")

    # making predictions and evaluate our model
    model_preds = model.predict(x_test)
    print("\nBaseline model evaluation")
    baseline_matrix = evaluate_model(y_test, model_preds)

    # set a grid of hyperperameters for finding the patterns with RandomizedSearchCV
    grid_rs = {
        "n_estimators": [10, 100, 200, 500, 1000, 1200],
        "max_depth": [None, 5, 10, 15, 20],
        "max_features": ['sqrt'],
        "min_samples_split": [2, 4, 6],
        "min_samples_leaf": [1, 2, 4]
    }

    # set a grid of hyperperameters for finding patterns with GridSearchCV
    grid_gs = {
        "n_estimators": [10, 100, 150],
        "max_depth": [None, 5],
        "max_features": ["sqrt"],
        "min_samples_split": [4, 6],
        "min_samples_leaf": [2, 3]
    }

    # Now we can improve our model with randomizedsearchcv and gridsearchcv
    rs_model, rs_matrix = improved_Randomizedsearvhcv(x_validation=x_val, y_validation=y_val,
                                                      x_test=x_test, y_test=y_test,
                                                      grid=grid_rs)
    gs_model, gs_matrix = improve_gridsearchcv(x_validation=x_val, y_validation=y_val,
                                               x_test=x_test, y_test=y_test,
                                               grid=grid_gs)

    # compare our models evaluations
    comp_matrix(baseline_matrix=baseline_matrix, rs_matrix=rs_matrix, gs_matrix=gs_matrix)

    # Now we can save our model with highest accuracy
    save_model(rs_model=rs_model, gs_model=gs_model, rs_matrix=rs_matrix, gs_matrix=gs_matrix)
