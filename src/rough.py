import pandas as pd

df = pd.read_csv('./data/combined_data.csv', low_memory=False)

print(df.collumns)