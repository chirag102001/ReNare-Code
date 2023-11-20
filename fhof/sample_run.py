from pathlib import Path

import pandas as pd

WORKDIR = Path(__file__).parent

# Load the CSV file into a pandas DataFrame
data = pd.read_csv(WORKDIR / 'random_data.csv')

# Display the imported data
print("Imported Data:")
print(data)

# Create a relationship between the parameters and calculate energy
data['Energy (Joules)'] = data['Number of Cells'] * data['Time (s)'] * data['Path Length (m)'] * data['Torque (Nm)']

# Display the updated DataFrame with the calculated energy
print("\nData with Energy Calculation:")
print(data)

# Save the DataFrame with the energy calculation to a new CSV file
data.to_csv(WORKDIR / 'data_with_energy.csv', index=False)

# Display the saved data
print("\nSaved Data with Energy:")
saved_data = pd.read_csv(WORKDIR / 'data_with_energy.csv')
print(saved_data)