# basic pandas operations

import pandas as pd

# create a dataframe
df = pd.DataFrame({
    'name': ['John', 'Jane', 'Jim', 'Jill'],
    'age': [25, 30, 35, 40],
    'city': ['New York', 'Los Angeles', 'Chicago', 'Houston']
})

# print the dataframe
print(df)

# merge, aggregate, and filter data
df_merged = pd.merge(df, df, on='name')
print(df_merged)

df_aggregated = df.groupby('city').agg({'age': 'mean'})
print(df_aggregated)

df_filtered = df[df['age'] > 30]
print(df_filtered)