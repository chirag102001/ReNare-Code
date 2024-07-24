import pandas as pd

# Load the CSV file into a pandas DataFrame
data = pd.read_csv('random_data.csv')

# Display the imported data
print("Imported Data:")
print(data)

# Create a relationship between the parameters and calculate energy
data['Energy (Joules)'] = data['Number of Cells'] * data['Time (s)'] * data['Path Length (m)'] * data['Torque (Nm)']

# Display the updated DataFrame with the calculated energy
print("\nData with Energy Calculation:")
print(data)

# Save the DataFrame with the energy calculation to a new CSV file
data.to_csv('data_with_energy.csv', index=False)

# Display the saved data
print("\nSaved Data with Energy:")
saved_data = pd.read_csv('data_with_energy.csv')
print(saved_data)