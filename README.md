# Predict Customer Churn

![](https://miro.medium.com/max/1400/1*lnPucWPldjHus0vFxruRTQ.png)

**Author:** Jorge Luis Ledesma Ureña

**Date:** April 2022

This documentation serves as a knowledge base for the **Predict Customer Churn** project of the ML DevOps Engineer Nanodegree Program at Udacity.

## 🚀 Project Description
This project is intended to identify credit card customers that are most likely to churn. It includes a Python package for a machine learning workflow that follows coding (PEP8) and engineering best practices for implementing software (modular, documented, and tested). The package also has the flexibility of being run interactively or from the command-line interface (CLI).

This project provided practice using skills for testing, logging, and best coding practices. It also introduced a problem data scientists across companies face all the time. How do we identify (and later intervene with) customers who are likely to churn?

## 📂 Files and data description
The data used for this project is based on the **Credit Cad customers** dataset from [Kaggle](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers). Here's an overview of the first rows:

| CLIENTUNUM | Customer_Age | Gender | Education_Level | Marital_Status | Income_Category |
|------------|--------------|--------|-----------------|----------------|-----------------|
| 768805383  | 45           | M      | High School     | Married        | 60K - 80K       |
| 818770008  | 49           | F      | Graduate        | Single         | Less than $40K  |
| 713982108  | 51           | M      | Graduate        | Married        | 80K - 120K      |
| 769911858  | 40           | F      | High School     | Unknown        | Less than $40K  |
| 709106358  | 40           | M      | Uneducated      | Married        | 60K - 80K       |

Here is the file structure of the project:

```
.
├── churn_notebook.ipynb
├── churn_library.py
├── churn_script_logging_and_tests.py
├── README.md
├── data             
├── images
│   ├── eda
│   └── results
├── logs
└── models
```

* **`churn_notebook.ipynb:`** Contains the code to be refactored.
* **`churn_library.py:`** Defines the functions for the machine learning workflow.
* **`churn_script_logging_and_tests.py:`** Defines the tests and logging process for the functions in `churn_library.py`.
* **`README.md:`** Provides project overview, and instructions to use the code.
* **`data:`** Stores the raw dataset `bank_data.csv`, input for the project.
* **`images:`** Stores image versions of results of EDA, model training and evaluation.
* **`logs:`** Stores logs, especially the `churn_library.log`, that contains the test results of the project.
* **`models:`** Stores executable models in `.plk` format.

## ✅ Running Files
To run the main library of functions to find customers who are likely to churn, we can proceed with any of the following CLI commands:
```
python churn_library.py
```
```
ipython churn_library.py
```
To run the unit tests for the `churn_library.py` functions and verify the log, we can proceed with any of the following CLI commands:
```
python churn_script_logging_and_tests.py
```
```
ipython churn_script_logging_and_tests.py
```
