'''
Churn library
---------------
Data Science processes including EDA, feature engineering,
model training, prediction, and model evaluation.

Author: Jorge Ledesma
Date: April 2022
'''

# import libraries
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report


os.environ['QT_QPA_PLATFORM']='offscreen'

logging.basicConfig(
    filename = './results.log',
    level = logging.INFO,
    filemode = 'w',
    format = '%(name)s - %(levelname)s: %(message)s')


def import_data(pth):
    '''
    Returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    try:
        logging.info('Input provided: %s', pth)
        assert isinstance(pth, str), 'The path must be of type str.'
        df = pd.read_csv(pth)
        logging.info('SUCCESS! Returning dataframe with shape %s.', df.shape)
        return df
    except AssertionError as msg:
        logging.error(msg)
    except FileNotFoundError:
        logging.error('No csv file found in the given path.')


def perform_eda(df):
    '''
    Perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    # Creating churn field
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    
    # Plotting churn distribution
    plt.figure(figsize=(20,10))
    fig = df['Churn'].hist();
    fig.get_figure.savefig('./images/eda/churn_distribution.png')
    
    # Plotting customer age distribution
    plt.figure(figsize=(20,10)) 
    fig = df['Customer_Age'].hist();
    fig.get_figure.savefig('./images/eda/customer_age_distribution.png')

    # Plotting marital status distribution
    plt.figure(figsize=(20,10)) 
    fig = df.Marital_Status.value_counts('normalize').plot(kind='bar');
    fig.get_figure.savefig('./images/eda/marital_status_distribution.png')
    
    # Plotting total transaction distribution
    plt.figure(figsize=(20,10))
    fig = sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True);
    fig.get_figure.savefig('./images/eda/total_transaction_distribution.png')
    
    # Plotting heatmap
    plt.figure(figsize=(20,10)) 
    fig = sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    fig.get_figure.savefig('./images/eda/total_transaction_distribution.png')
    
def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    pass


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    pass


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    pass

def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    pass