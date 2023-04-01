import pandas as pd
import numpy as np
df = pd.read_csv('adultDs.csv')


print(df)
print(f'dataset shape is {df.shape}')
print(df.describe())
