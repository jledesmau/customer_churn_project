'''
Testing functions for churn_library.py

Author: Jorge Ledesma
Date: April 2022
'''

import os
import shutil
import logging
import pandas as pd
import churn_library as chlib

# Creating missing directories
if not os.path.exists("./logs/"):
    os.mkdir("./logs/")

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')

# Constants
DATA_PTH = "./data/bank_data.csv"
DATA_DF = pd.read_csv(DATA_PTH)
DATA_DF['Churn'] = DATA_DF['Attrition_Flag'].apply(
    lambda val: 0 if val == "Existing Customer" else 1)
CATEGORY_LST = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category']
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = chlib.perform_feature_engineering(DATA_DF)


def test_import():
    '''
    test data import
    '''
    error_count = 0
    logging.info("START: Testing import_data")

    try:
        data_df = chlib.import_data(DATA_PTH)
        shape = data_df.shape
        logging.info("Shape of dataset is %s", shape)
        assert (shape[0] > 0 and shape[1] > 0)
    except FileNotFoundError:
        logging.error("The file wasn't found")
        error_count += 1
    except AssertionError:
        logging.error("The file doesn't have rows or columns")
        error_count += 1

    if error_count:
        logging.error(
            "DONE: Testing import_data, %s errors found",
            error_count)
    else:
        logging.info("SUCCESS: Testing import_data, 0 errors found")


def test_eda():
    '''
    test perform eda function
    '''
    # Deleting path to start form scratch
    if os.path.exists("./images/eda/"):
        shutil.rmtree("./images/eda/")

    error_count = 0
    logging.info("START: Testing perform_eda")

    chlib.perform_eda(DATA_DF)

    graphs = [
        'churn',
        'customer_Age',
        'marital_status',
        'total_transaction',
        'heatmap']

    for graph in graphs:
        if graph == 'heatmap':
            name = 'heatmap.png'
        else:
            name = f"{graph.lower()}_distribution.png"

        try:
            assert os.path.exists(f"./images/eda/{name}")
            logging.info("Plot saved in ./images/eda/%s", name)
        except AssertionError:
            logging.error("%s is missing", name)

    if error_count:
        logging.error(
            "DONE: Testing perform_eda, %s errors found",
            error_count)
    else:
        logging.info("SUCCESS: Testing perform_eda, 0 errors found")


def test_encoder_helper():
    '''
    test encoder helper
    '''
    try:
        logging.info("START: Testing encoder_helper")
        encoded_df = chlib.encoder_helper(DATA_DF, CATEGORY_LST)
        shape = encoded_df.shape
        assert (shape[0] > 0 and shape[1] > 0)
        logging.info("SUCCESS: Testing encoder_helper, 0 errors found")
    except AssertionError:
        logging.error("The dataframe doesn't have rows or columns")
        logging.error("DONE: Testing encoder_helper, 1 error found")


def test_feature_engineering():
    '''
    test perform_feature_engineering
    '''
    error_count = 0
    logging.info("START: Testing perform_feature_engineering")

    data_split = chlib.perform_feature_engineering(DATA_DF)

    labels = ['x_train', 'x_test', 'y_train', 'y_test']

    for i in range(4):
        shape = data_split[i].shape
        logging.info("Shape of %s is %s", labels[i], shape)
        try:
            if i < 2:
                assert (shape[0] > 0 and shape[1] > 0)
            else:
                assert shape[0] > 0
        except AssertionError:
            logging.error("%s has no values", labels[i])
            error_count += 1
    try:
        assert data_split[0].shape[0] == data_split[2].shape[0]
    except AssertionError:
        logging.error("Number of rows in x_train and y_train are different")
        error_count += 1

    try:
        assert data_split[1].shape[0] == data_split[3].shape[0]
    except AssertionError:
        logging.error("Number of rows in x_test and y_test are different")
        error_count += 1

    if error_count:
        logging.error(
            "DONE: Testing perform_feature_engineering, %s errors found",
            error_count)
    else:
        logging.info(
            "SUCCESS: Testing perform_feature_engineering, 0 errors found")


def test_train_models():
    '''
    test train_models
    '''
    # Deleting paths to start form scratch
    if os.path.exists("./images/results/"):
        shutil.rmtree("./images/results/")
    if os.path.exists("./models/"):
        shutil.rmtree("./models")

    error_count = 0
    logging.info("START: Testing train_models")

    chlib.train_models(X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)

    results = [
        'random_forest_results.png',
        'logistic_regression_results.png',
        'roc_curve_result.png',
        'feature_importances.png']

    for name in results:
        try:
            assert os.path.exists(f"./images/results/{name}")
            logging.info("Result saved in ./images/results/%s", name)
        except AssertionError:
            logging.error("%s is missing", name)
            error_count += 1

    try:
        assert os.path.exists("./models/rfc_model.pkl")
        logging.info("Random Forest model saved in ./models/rfc_model.pkl")
    except AssertionError:
        logging.error("Random Forest model is missing")
        error_count += 1

    try:
        assert os.path.exists("./models/logistic_model.pkl")
        logging.info(
            "Logistic Regression model saved in ./models/logistic_model.pkl")
    except AssertionError:
        logging.error("Logistic Regression model is missing")
        error_count += 1

    if error_count:
        logging.error(
            "DONE: Testing train_models, %s errors found",
            error_count)
    else:
        logging.info("SUCCESS: Testing train_models, 0 errors found")


if __name__ == "__main__":
    test_import()
    test_eda()
    test_encoder_helper()
    test_feature_engineering()
    test_train_models()
