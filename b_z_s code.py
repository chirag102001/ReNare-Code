from pathlib import Path

import pandas as pd
import numpy as np

WORKDIR = Path(__file__).parent

# Step 1: Import the CSV file containing velocity data
velocity_df = pd.read_csv(WORKDIR / 'data_sensordata.csv',low_memory=False)

inertial_df = pd.read_csv(WORKDIR / "parameters_inertial.csv", index_col="body_name", header=0)
link_ids = inertial_df.index.to_list()

link_id = link_ids[1]  # should be robot_link_1
link_id_second = link_ids[2] # robot_link_2

# Step 2: Extract the required columns for linear and angular velocities
linvel_columns = [f'{link_id}_linvel.{axis}' for axis in "xyz"]
angvel_columns = [f'{link_id}_angvel.{axis}' for axis in "xyz"]

linvel_df = velocity_df[linvel_columns]
angvel_df = velocity_df[angvel_columns]

# # Step 3: Import the CSV file containing inertia data for multiple robot links
# inertia_df = pd.read_csv('parameters_inertial.csv')

# # Step 4: Filter the inertia data for robot link 1
# link1_inertia_df = inertia_df[inertia_df['inertia.xx'] == 'robot_link_1']


# # Step 5: Get the mass and inertia data for link 1
# link1_mass = link1_inertia_df['mass'].values[0]
# link1_inertial = np.array(link1_inertia_df[['inertia_xx', 'inertia_yy', 'inertia_zz']])

link1_mass = inertial_df.loc[link_id]["mass"]
link1_inertial = np.diag(inertial_df.loc[link_id].loc[[f"inertia.{axis}" for axis in ("xx", "yy", "zz")]])

def calculate_kinetic_energy(row):
    linvel_values = np.array(row[linvel_columns])
    angvel_values = np.array(row[angvel_columns])

    kinetic_energy = (
        (link1_mass * np.dot(linvel_values, linvel_values) / 2) +
        (np.dot(angvel_values, np.dot(link1_inertial, angvel_values)) / 2)
    )

    return kinetic_energy

# Step 5: Apply the calculate_kinetic_energy function to each row and create a new column for the results
velocity_df['kinetic_energy'] = velocity_df.apply(calculate_kinetic_energy, axis=1)

# Step 6: Calculate the total kinetic energy by summing all row values
absolute_power = velocity_df["kinetic_energy"].diff(1).abs()
total_kinetic_energy = absolute_power.sum() * velocity_df["time"].mean()

# Step 7: You can now access the total kinetic energy value
print("Total Kinetic Energy:", total_kinetic_energy)