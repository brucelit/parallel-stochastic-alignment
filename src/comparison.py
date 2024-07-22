import pandas as pd

# Load the CSV files into DataFrames
df1 = pd.read_csv("../src/stochastic_astar.csv")
df2 = pd.read_csv("../src/non_stochastic_astar.csv")

# Select and sort the columns to compare (assuming they have a common index column)
column1 = df1[['cost']]
column2 = df2[['cost']]

# Compare the columns element-wise
comparison = column1['cost'].equals(column2['cost'])

# Print the result
if comparison:
    print("The columns are equal in every cell.")
else:
    print("The columns are not equal in every cell.")
