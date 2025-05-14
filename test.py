import pandas as pd

df = pd.read_csv('data/sampled_labeled.csv')
df_sample = df.sample(1000)  # or head(1000)
df_sample.to_csv('data/sampled_labeled_small.csv', index=False)
