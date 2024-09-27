import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2 calculate BMI
df['bmi']=round(df.weight / ((df.height/100)**2),2)

# 3 Normalize data by making 0 always good and 1 always bad. If the value of cholesterol or gluc is 1, set the value to 0. If the value is more than 1, set the value to 1.
df['overweight'] = np.where(df['bmi'].values > 25, 1, 0)
df = df.drop('bmi', axis=1)
df['cholesterol'] = np.where(df['cholesterol'].values > 1, 1, 0)
df['gluc'] = np.where(df['gluc'].values > 1, 1, 0)
df['smoke'] = np.where(df['smoke'].values >= 1, 1, 0)
df['alco'] = np.where(df['alco'].values >= 1, 1, 0)
df['active'] = np.where(df['active'].values >= 1, 1, 0)

# 4
def draw_cat_plot():
  # 5 Create a DataFrame for the cat plot using pd.melt with values from cholesterol, gluc, smoke, alco, active, and overweight in the df_cat variable.
  df_cat= pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol','gluc','smoke','alco','active','overweight'])

  # 6 Group and reformat the data in df_cat to split it by cardio. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
  df_cardio = df_cat.groupby(['cardio','variable','value']).agg({'value': ['count']}).reset_index()
  #rename columns 
  df_cardio.columns = ['cardio','variable','value', 'total']

  # 7 Convert the data into long format and create a chart that shows the value counts of the categorical features using the following method provided by the seaborn library 
  fig, ax = plt.subplots()
  fig=sns.catplot(kind='bar', data=df_cardio, x='variable', y='total', hue='value', col='cardio').fig

  # 8 Get the figure for the output and store it in the fig variable
  #fig.tight_layout()

  # 9
  fig.savefig('catplot.png')
  return fig


# 10
def draw_heat_map():
  
  # 11
  #Clean the data. Filter out the following patient segments that represent incorrect data:
  #diastolic pressure is higher than systolic (Keep the correct data with (df['ap_lo'] <= df['ap_hi']))
  df_heat = None
  flt_pressure= df['ap_lo'] <= df['ap_hi']
  #height is less than the 2.5th percentile (Keep the correct data with (df['height'] >= df['height'].quantile(0.025)))
  flt_height_ge25= df['height'] >= df['height'].quantile(0.025)
  #height is more than the 97.5th percentile
  flt_height_le97= df['height'] <= df['height'].quantile(0.975)
  #weight is less than the 2.5th percentile
  flt_weight_ge25= df['weight'] >= df['weight'].quantile(0.025)
  #weight is more than the 97.5th percentile
  flt_weight_le97= df['weight'] <= df['weight'].quantile(0.975)

  df_heat = df.loc[(flt_pressure & flt_height_ge25 & flt_height_le97 & flt_weight_ge25 & flt_weight_le97) ]
  #.rename(columns={'sex':'gender'})

  # 12 Calculate the correlation matrix and store it in the corr variable
  corr = df_heat.corr(method='pearson')

  # 13 Generate a mask for the upper triangle and store it in the mask variable
  mask = np.triu(np.ones_like(corr)).astype(bool)

  # 14 Set up the matplotlib figure
  fig, ax = plt.subplots()

  # 15 Plot the correlation matrix using the method provided by the seaborn library import: sns.heatmap()
  ax=sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", annot_kws={"size":7})
  
  fig.savefig('heatmap.png')
  return fig
