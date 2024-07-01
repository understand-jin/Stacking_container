import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('relocation_count.csv')

# Clean up the dataframe by renaming columns
df.columns = ['Scenario', 'Empty1', 'MIP_total_relocation', 'MIP_mean_relocation',
              'Heuristic1_total_relocation', 'Heuristic1_mean_relocation',
              'Heuristic2_total_relocation', 'Heuristic2_mean_relocation']
df = df.drop([0]).reset_index(drop=True)  # Remove the first row which was column headers

# Remove unnecessary column
df = df.drop(columns=['Empty1'])

# Strip any leading/trailing spaces from Scenario column
df['Scenario'] = df['Scenario'].str.strip()

# Convert columns to numeric where possible
for col in df.columns[1:]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Plot the mean relocations for each method
plt.figure(figsize=(10, 6))

plt.plot(df['Scenario'], df['MIP_mean_relocation'], marker='o', label='MIP Mean Relocation')
plt.plot(df['Scenario'], df['Heuristic1_mean_relocation'], marker='s', label='Heuristic1 Mean Relocation')
plt.plot(df['Scenario'], df['Heuristic2_mean_relocation'], marker='^', label='Heuristic2 Mean Relocation')

plt.title('Mean Relocations for Each Scenario')
plt.xlabel('Scenario',fontsize = 15,  labelpad = 15)
plt.ylabel('Mean Relocations', fontsize = 15, labelpad = 15)
plt.xticks(rotation=0, fontsize = 10)
plt.yticks(rotation = 0, fontsize = 10)
plt.legend()

plt.tight_layout()
plt.show()
