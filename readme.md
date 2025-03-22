Here’s the corrected and properly formatted Markdown:  

# Linear Regression with Python

This Jupyter Notebook demonstrates a simple implementation of linear regression using Python. The goal is to predict salaries based on years of experience using a dataset. The notebook covers data loading, visualization, model training, evaluation, and visualization of the results.

## Table of Contents
1. [Introduction](#introduction)  
2. [Dependencies](#dependencies)  
3. [Data Loading](#data-loading)  
4. [Data Visualization](#data-visualization)  
5. [Model Training](#model-training)  
6. [Model Evaluation](#model-evaluation)  
7. [Results Visualization](#results-visualization)  
8. [R-squared Score](#r-squared-score)  
9. [Conclusion](#conclusion)  

## Introduction
Linear regression is a statistical method that models the relationship between a dependent variable and one or more independent variables. In this notebook, we use linear regression to predict salaries based on years of experience.

## Dependencies
The following Python libraries are used in this notebook:  
- `pandas`: For data manipulation and analysis.  
- `matplotlib`: For data visualization.  
- `scikit-learn`: For machine learning tasks, including model training and evaluation.  

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
```

## Data Loading
The dataset `salaries.csv` is loaded using pandas. The dataset contains two columns: `years_of_experience` and `salary`.  

```python
df = pd.read_csv('salaries.csv')
df.head()
```

## Data Visualization
The relationship between years of experience and salary is visualized using a scatter plot.  

```python
plt.scatter(df['years_of_experience'], df['salary'])
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Years of Experience vs Salary')
plt.show()
```

## Model Training
The dataset is split into training and testing sets. A linear regression model is then trained on the training data.  

```python
x = df[['years_of_experience']]
y = df['salary']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(x_train, y_train)
```

## Model Evaluation
The trained model is used to make predictions on the test data. The predictions are compared with the actual values to evaluate the model's performance.  

```python
y_pred = model.predict(x_test)
error = y_pred - y_test
```

## Results Visualization
The results are visualized by plotting the actual values against the predicted values.  

```python
plt.scatter(x_test, y_test, label='Actual')
plt.plot(x_test, y_pred, color='yellow', label='Predicted')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Actual vs Predicted Salaries')
plt.legend()
plt.show()
```

## R-squared Score
The R-squared score is calculated to measure the model's accuracy. The R-squared score ranges from 0 to 1, where 1 indicates a perfect fit.  

```python
r2 = r2_score(y_test, y_pred)
print(f'R-squared Score: {r2:.4f}')
```

## Conclusion
This notebook provides a basic implementation of linear regression to predict salaries based on years of experience. The model's performance is evaluated using the R-squared score, and the results are visualized to understand the relationship between the variables.

