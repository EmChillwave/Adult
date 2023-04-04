import pandas as pd
import numpy as np


def get_dateFrame(filename):
    df = pd.read_csv((filename))
    return df


salaries = 'adultDs'
df = get_dateFrame('{}.csv'.format(salaries))
print(df)
print('Hours Per Week statistics')
print('Mean: {} - Standard Deviation: {} - Median: {}'.format(
    df['Hours per week'].mean(), df['Hours per week'].std(), df['Hours per week'].median()))
print('Age Statistics')
print('Mean: {} - Standard Deviation: {} - Median: {}'.format(
    df['Age'].mean(), df['Age'].std(), df['Age'].median()))
