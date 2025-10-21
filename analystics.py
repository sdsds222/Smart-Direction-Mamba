import pandas as pd
df=pd.read_csv('training_data.csv')
print(f'总样本: {len(df)}')
print(df['direction'].value_counts())