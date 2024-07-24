import pandas as pd
import numpy as np
from pathlib import Path

# Define the working directory
WORKDIR = Path(__file__).parent

# Step 1: Import the CSV file containing velocity data
velocity_df = pd.read_csv(WORKDIR / 'data_sensordata.csv', low_memory=False)

# Step 2: Import the CSV file containing inertia data for multiple robot links
inertial_df = pd.read_csv(WORKDIR / "parameters_inertial.csv", index_col="body_name", header=0)

# Get the list of link IDs
link_ids = inertial_df.index.to_list()

# Loop through the first six links
for i in range(1, 7):
    # Extract the required columns for linear and angular velocities for the current link
    link_id = link_ids[i]  # Adjust the index to start from 0
    linvel_columns = [f'{link_id}_linvel.{axis}' for axis in "xyz"]
    angvel_columns = [f'{link_id}_angvel.{axis}' for axis in "xyz"]

    linvel_df = velocity_df[linvel_columns]
    angvel_df = velocity_df[angvel_columns]

    # Get the mass and inertia data for the current link
    link_mass = inertial_df.loc[link_id]["mass"]
    link_inertial = np.diag(inertial_df.loc[link_id].loc[[f"inertia.{axis}" for axis in ("xx", "yy", "zz")]])

    # Define a function to calculate kinetic energy for the current link
    def calculate_kinetic_energy(row):
        linvel_values = np.array(row[linvel_columns])
        angvel_values = np.array(row[angvel_columns])

        kinetic_energy = (
            (link_mass * np.dot(linvel_values, linvel_values) / 2) +
            (np.dot(angvel_values, np.dot(link_inertial, angvel_values)) / 2)
        )

        return kinetic_energy

    # Apply the calculate_kinetic_energy function to each row and create a new column for the current link
    velocity_df[f'kinetic_energy_link_{i}'] = velocity_df.apply(calculate_kinetic_energy, axis=1)

    # Calculate the total kinetic energy for the current link by summing all row values
    absolute_power = velocity_df[f'kinetic_energy_link_{i}'].diff(1).abs()
    total_kinetic_energy = absolute_power.sum() * velocity_df["time"].mean()

    # Display the total kinetic energy for the current link
    print(f"Total Kinetic Energy (Link {i}):", total_kinetic_energy)