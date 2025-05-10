# Titanic Dataset Analysis

This repository contains a comprehensive analysis of the Titanic dataset, focusing on data exploration, preprocessing, and the implementation of a machine learning model using logistic regression. The model achieved an accuracy of 80% in predicting passenger survival.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Analysis Steps](#analysis-steps)
- [Machine Learning Model](#machine-learning-model)
- [Results](#results)
- [Conclusion](#conclusion)
- [License](#license)

## Introduction

The Titanic dataset is a well-known dataset used for binary classification tasks. The goal of this analysis is to predict whether a passenger survived the Titanic disaster based on various features such as age, gender, class, and fare.

## Dataset

The dataset used for this analysis is the Titanic dataset, which can be found on [Kaggle](https://www.kaggle.com/c/titanic). The dataset includes the following features:
- `PassengerId`: Unique identifier for each passenger
- `Pclass`: Ticket class (1st, 2nd, or 3rd)
- `Name`: Name of the passenger
- `Sex`: Gender of the passenger
- `Age`: Age of the passenger
- `SibSp`: Number of siblings/spouses aboard
- `Parch`: Number of parents/children aboard
- `Ticket`: Ticket number
- `Fare`: Fare paid for the ticket
- `Embarked`: Port of embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
- `Survived`: Survival status (0 = No, 1 = Yes)

## Analysis Steps

1. **Data Loading**: Load the Titanic dataset using Pandas.
2. **Data Exploration**: Explore the dataset to understand its structure and identify missing values.
3. **Data Cleaning**: Handle missing values and perform necessary data transformations.
4. **Feature Engineering**: Create new features that may improve model performance (e.g., extracting titles from names).
5. **Data Visualization**: Visualize the data to identify patterns and relationships between features and survival.
6. **Data Splitting**: Split the dataset into training and testing sets.
7. **Model Implementation**: Implement a logistic regression model using Scikit-learn.
8. **Model Evaluation**: Evaluate the model's performance using accuracy and other metrics.

## Machine Learning Model

The logistic regression model was implemented using the Scikit-learn library. The following steps were taken:

1. **Import Libraries**:
   ```python
   import pandas as pd
   import numpy as np
   from sklearn.model_selection import train_test_split
   from sklearn.linear_model import LogisticRegression
   from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
