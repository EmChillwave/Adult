import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder


def get_dateFrame(filename):
    df = pd.read_csv(('adultDs.csv'))
    return df

# Mean, Standard Deviation, etc.


def descriptions():
    rdescription = df['Race'].describe()
    HPW_description = df['Hours per week'].describe()
    age_description = df['Age'].describe()
    print(rdescription)
    print(HPW_description)
    print(age_description)


df = pd.read_csv('adultDS.csv')
print(df)
descriptions()

# Histogram of Hours per week
plt.hist(df['Hours per week'], bins=20)
plt.title('Histogram of Hours per week')
plt.xlabel('Hours per week')
plt.ylabel('Frequency')
plt.show()

# Boxplot of Age
plt.boxplot(df['Age'])
plt.title('Boxplot of Age')
plt.ylabel('Age')
plt.show()

# Define the input and target variables
X = df[['Age', 'Hours per week']]
y = df['Capital Gain']

# One-hot encode the 'Occupation' and 'Education level' columns
ohe = OneHotEncoder(sparse=False)
cat_cols = ['Occupation', 'Education']
X_cat = ohe.fit_transform(df[cat_cols])

# Concatenate the one-hot encoded columns with the numerical columns
X = pd.concat(
    [X, pd.DataFrame(X_cat, columns=ohe.get_feature_names_out(cat_cols))], axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Create the model and fit it to the training data
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict the income level on the testing data
y_pred = model.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
