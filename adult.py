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
    education_description = df['Education'].describe()
    print(rdescription)
    print(HPW_description)
    print(age_description)
    print(education_description)


df = pd.read_csv('adultDS.csv')
print(df)
descriptions()

# Bar chart of Education level vs Count
plt.figure(figsize=(10, 5))
df['Education'].value_counts().plot(kind='bar')
plt.title('Education Level Distribution')
plt.xlabel('Education Level')
plt.ylabel('Count')
plt.show()

# Scatter plot of Age vs Capital Gain
plt.figure(figsize=(10, 5))
plt.scatter(df['Age'], df['Capital Gain'], alpha=0.5)
plt.title('Age vs Capital Gain')
plt.xlabel('Age')
plt.ylabel('Capital Gain')
plt.show()

# Create a pivot table to count the number of records for each combination of Race and Occupation
pivot_table = pd.pivot_table(df, values='Age', index=['Race'], columns=[
                             'Occupation'], aggfunc='count')

# Plot the line chart
pivot_table.plot(kind='line', marker='o')

# Set the title and axis labels
plt.title('Counts by Race and Occupation')
plt.xlabel('Race')
plt.ylabel('Counts')

# Show the chart
plt.show()

# Box plot of Income vs Education level
plt.figure(figsize=(10, 5))
df.boxplot(column=['Capital Gain'], by='Education', rot=45)
plt.title('Income vs Education Level')
plt.xlabel('Education Level')
plt.ylabel('Capital Gain')
plt.show()

# REGRESSION MODEL TO CALCULATE ACCURACY
df = pd.read_csv('adultDs.csv')
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
