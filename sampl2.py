import pandas as pd
import numpy as np

# Step 1: Import the CSV file containing velocity data
velocity_df = pd.read_csv('data_sensordata.csv',low_memory=False)

# Step 2: Extract the required columns for linear and angular velocities
linvel_columns = ['robot_link_1_linvel.x', 'robot_link_1_linvel.y', 'robot_link_1_linvel.z']
angvel_columns = ['robot_link_1_angvel.x', 'robot_link_1_angvel.y', 'robot_link_1_angvel.z']

linvel_df = velocity_df[linvel_columns]
angvel_df = velocity_df[angvel_columns]

# # Step 3: Import the CSV file containing inertia data for multiple robot links
# inertia_df = pd.read_csv('parameters_inertial.csv')

# # Step 4: Filter the inertia data for robot link 1
# link1_inertia_df = inertia_df[inertia_df['inertia.xx'] == 'robot_link_1']


# # Step 5: Get the mass and inertia data for link 1
# link1_mass = link1_inertia_df['mass'].values[0]
# link1_inertial = np.array(link1_inertia_df[['inertia_xx', 'inertia_yy', 'inertia_zz']])

link1_mass = 7.778
link1_inertial = np.array([[0.0314743, 0, 0],
                           [0, 0.0314743, 0],
                           [0, 0, 0.0218756]])

def calculate_kinetic_energy(row):
    linvel_values = np.array(row[linvel_columns])
    angvel_values = np.array(row[angvel_columns])
# (mass * transpose(linvel) * linvel / 2 + transpose(angvel) * inertial * angvel / 2)
    kinetic_energy = (
        (link1_mass * np.dot(linvel_values, linvel_values) / 2) +
        (np.dot(angvel_values, np.dot(link1_inertial, angvel_values)) / 2)
    )
    
    return kinetic_energy

# Step 5: Apply the calculate_kinetic_energy function to each row and create a new column for the results
velocity_df['kinetic_energy'] = velocity_df.apply(calculate_kinetic_energy, axis=1)

# Step 6: Calculate the total kinetic energy by summing all row values
total_kinetic_energy = velocity_df['kinetic_energy'].sum()

# Step 7: You can now access the total kinetic energy value
print("Total Kinetic Energy:", total_kinetic_energy)