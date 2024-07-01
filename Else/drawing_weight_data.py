import pandas as pd
import matplotlib.pyplot as plt

# Load the new CSV file
file_path = 'weight_count.csv'
df_new = pd.read_csv(file_path)

# Display the columns to understand the structure
print(df_new.columns)
print(df_new.head())

# Based on the error message and structure, set the correct column names
df_new.columns = ['Scenario', 'Empty1', 'Heuristic1_total_relocation', 'Heuristic1_mean_relocation',
                  'Heuristic2_total_relocation', 'Heuristic2_mean_relocation', 'Empty2', 'Container_count',
                  'Empty3', 'MIP', 'Empty4', 'MIP(alpha 0.5, beta 0.5)_total_relocation', 'MIP(alpha 0.5, beta 0.5)_mean_relocation',
                  'MIP(alpha 0, beta 1)_mean_relocation', 'Empty5']
df_new = df_new.drop([0]).reset_index(drop=True)  # Remove the first row which was column headers

# Remove unnecessary columns
df_new = df_new.drop(columns=['Empty1', 'Empty2', 'Empty3', 'Empty4', 'Empty5'])

# Strip any leading/trailing spaces from Scenario column
df_new['Scenario'] = df_new['Scenario'].str.strip()

# Remove rows where 'Scenario' is NaN
df_new = df_new.dropna(subset=['Scenario'])

# Convert columns to numeric where possible
for col in df_new.columns[1:]:
    df_new[col] = pd.to_numeric(df_new[col], errors='coerce')

# Plot the mean relocations for each method
plt.figure(figsize=(14, 7))

plt.plot(df_new['Scenario'], df_new['Heuristic1_mean_relocation'], marker='s', label='Heuristic1 Mean Relocation')
plt.plot(df_new['Scenario'], df_new['Heuristic2_mean_relocation'], marker='^', label='Heuristic2 Mean Relocation')
# plt.plot(df_new['Scenario'], df_new['MIP(alpha 0.5, beta 0.5)_mean_relocation'], marker='o', label='MIP(alpha 0.5, beta 0.5) Mean Relocation')
# plt.plot(df_new['Scenario'], df_new['MIP(alpha 0, beta 1)_mean_relocation'], marker='o', label='MIP(alpha 0, beta 1) Mean Relocation')

# Plot specific points for MIP(alpha 0, beta 1)_mean_relocation and MIP(alpha 0, beta 1) Mean Relocation
plt.scatter(df_new['Scenario'][0], 3.75, color='red', zorder=5, label='MIP(alpha 0.5, beta 0.5)')
plt.scatter(df_new['Scenario'][0], 19.96, color='blue', zorder=5, label='MIP(alpha 0, beta 1)')

plt.title('Mean Weight Violation for Each Scenario')
plt.xlabel('Scenario', labelpad=15, fontsize=15)  # Add label padding
plt.ylabel('Mean weight violation', labelpad=15, fontsize=15)  # Add label padding
plt.xticks(rotation=0, fontsize=10)
plt.yticks(rotation=0, fontsize=10)  # Ensure labels are not rotated
plt.legend()

plt.tight_layout()
plt.show()

