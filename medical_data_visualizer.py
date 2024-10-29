import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Ensure plots display inline in Jupyter
%matplotlib inline

# Step 1: Load the data
df = pd.read_csv('')

# Step 2: Add 'overweight' column
df['overweight'] = ((df['weight'] / ((df['height'] / 100) ** 2)) > 25).astype(int)

# Step 3: Normalize cholesterol and glucose values
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# Step 4: Categorical Plot Function
def draw_cat_plot():
    # Melt the DataFrame for categorical plotting
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # Group by cardio, variable, and value to get counts
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # Draw the categorical plot
    fig = sns.catplot(x='variable', y='total', hue='value', col='cardio', data=df_cat, kind='bar')
    
    # Show the plot in Jupyter Notebook
    plt.show()
    return fig

# Step 5: Heat Map Function
def draw_heat_map():
    # Clean the data by filtering out incorrect data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Draw the heatmap
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', center=0, square=True, cbar_kws={"shrink": 0.5}, ax=ax)
    
    # Show the plot in Jupyter Notebook
    plt.show()
    return fig

# Generate and display the categorical plot
draw_cat_plot()

# Generate and display the heatmap plot
draw_heat_map()
