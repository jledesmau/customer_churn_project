'''
Churn library
---------------
Data Science processes including EDA, feature engineering,
model training, prediction, and model evaluation.

Author: Jorge Ledesma
Date: April 2022
'''

# import libraries
import os
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    Returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            data_df: pandas dataframe
    '''

    data_df = pd.read_csv(pth)
    return data_df


def perform_eda(data_df):
    '''
    Perform EDA on data_df and save figures to images folder

    input:
            data_df: pandas dataframe
    output:
            None
    '''

    # Creating missing directories
    if not os.path.exists("./images/"):
        os.mkdir("./images/")
    if not os.path.exists("./images/eda/"):
        os.mkdir("./images/eda/")

    # Creating churn field
    data_df['Churn'] = data_df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # Naming graphs
    graphs = [
        'Churn',
        'Customer_Age',
        'Marital_Status',
        'Total_Trans_Ct',
        'heatmap']

    for name in graphs:
        plt.figure(figsize=(20, 10))

        # Ploting each kind of graph
        if name in ('Churn', 'Customer_Age'):
            fig = data_df[name].plot.hist()
        elif name == 'Marital_Status':
            fig = data_df[name].value_counts('normalize').plot.bar()
        elif name == 'Total_Trans_Ct':
            fig = sns.displot(data_df[name], stat='density', kde=True)
            name = 'Total_Transaction'
        else:
            fig = sns.heatmap(
                data_df.corr(),
                annot=False,
                cmap='Dark2_r',
                linewidths=2)

        # Saving graphs as image files
        if name != 'heatmap':
            fig.figure.savefig(f'./images/eda/{name.lower()}_distribution.png')
        else:
            fig.figure.savefig('./images/eda/heatmap.png')


def encoder_helper(data_df, category_lst):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            data_df: pandas dataframe
            category_lst: list of columns that contain categorical features
    output:
            data_df: pandas dataframe with new columns for categorical variables
    '''
    for category in category_lst:
        category_groups = data_df.groupby(category).mean()['Churn']
        data_df[f'{category}_Churn'] = [category_groups.loc[val]
                                        for val in data_df[category]]

    return data_df


def perform_feature_engineering(data_df):
    '''
    Performs feature engineering and train-test split to data_df
    input:
              data_df: pandas dataframe
    output:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    cat_columns = ['Gender',
                   'Education_Level',
                   'Marital_Status',
                   'Income_Category',
                   'Card_Category']

    encoded_data_df = encoder_helper(data_df, cat_columns)

    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    target = data_df['Churn']
    features = pd.DataFrame()
    features[keep_cols] = encoded_data_df[keep_cols]
    data_df_split = train_test_split(
        features, target, test_size=0.3, random_state=42)

    return data_df_split


def classification_report_image(y_train,
                                y_test,
                                y_train_test_preds_lr,
                                y_train_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_test_preds_lr: list of training and test predictions from logistic regression
            y_train_test_preds_rf: list of training and test predictions from random forest
    output:
             None
    '''
    # Creating missing directories
    if not os.path.exists("./images/"):
        os.mkdir("./images/")
    if not os.path.exists("./images/results/"):
        os.mkdir("./images/results/")

    models = {'Random Forest': y_train_test_preds_rf,
              'Logistic Regression': y_train_test_preds_lr}

    for name, y_train_test in models.items():
        plt.figure(figsize=(6, 5))
        plt.rc('font', family='monospace')

        plt.text(0.01, 1.05, str(f'{name} Train'))
        plt.text(
            0.00,
            1.00,
            str('--------------------------------------------------------'))
        plt.text(
            0.01, 0.60, str(
                classification_report(
                    y_train, y_train_test[0])))
        plt.text(0.01, 0.50, str(f'{name} Test'))
        plt.text(
            0.00,
            0.45,
            str('--------------------------------------------------------'))
        plt.text(
            0.01, 0.05, str(
                classification_report(
                    y_test, y_train_test[1])))
        plt.axis('off')
        plt.savefig(
            f'./images/results/{name.lower().replace(" ", "_")}_results.png')


def roc_plot_image(models, x_test, y_test):
    '''
    Creates and stores ROC Curve results for the modelsthe feature importances in pth

    input:
            models: list of models to be plotted
            x_test: X testing data
            y_test: test response values
    output:
             None
    '''
    # Creating missing directories
    if not os.path.exists("./images/"):
        os.mkdir("./images/")
    if not os.path.exists("./images/results/"):
        os.mkdir("./images/results/")

    plt.figure(figsize=(15, 8))
    axes = plt.gca()

    for model in models:
        plot_roc_curve(model, x_test, y_test, ax=axes, alpha=0.8)

    plt.savefig('./images/results/roc_curve_result.png')


def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            x_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Sort feature importances in descending order
    indices = np.argsort(model)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    plt.bar(range(x_data.shape[1]), model[indices])
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth, bbox_inches="tight")


def train_models(x_train, x_test, y_train, y_test):
    '''
    Train, store model results: images + scores, and store models
    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # Grid search
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)

    # Models training
    cv_rfc.fit(x_train, y_train)
    lrc.fit(x_train, y_train)

    # Prediction
    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    # Model Evaluation
    classification_report_image(y_train,
                                y_test,
                                [y_train_preds_lr, y_test_preds_lr],
                                [y_train_preds_rf, y_test_preds_rf])

    roc_plot_image([cv_rfc.best_estimator_, lrc],
                   x_test,
                   y_test)

    importances = cv_rfc.best_estimator_.feature_importances_
    x_data = pd.concat([x_train, x_test], axis=0).reset_index(drop=True)
    output_pth = './images/results/feature_importances.png'
    feature_importance_plot(importances, x_data, output_pth)

    # Creating missing directories
    if not os.path.exists("./models/"):
        os.mkdir("./models/")

    # Saving models
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')


if __name__ == "__main__":
    FILE_PATH = "./data/bank_data.csv"
    DATA_DF = import_data(FILE_PATH)
    perform_eda(DATA_DF)
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = perform_feature_engineering(DATA_DF)
    #train_models(X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)
