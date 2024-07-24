import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px

# Define the working directory
WORKDIR = Path(__file__).parent

# Step 1: Import the CSV file containing velocity data
velocity_df = pd.read_csv(WORKDIR / 'data_sensordata.csv', low_memory=False)

# Step 2: Import the CSV file containing inertia data for multiple robot links
inertial_df = pd.read_csv(WORKDIR / "parameters_inertial.csv", index_col="body_name", header=0)

# Get the list of link IDs
link_ids = inertial_df.index.to_list()

# Initialize an empty DataFrame to store kinetic energy data for all links
kinetic_energy_df = pd.DataFrame()

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

    # Filter rows based on the "step" column (replace 'prepositioning' with the actual value in your dataset)
    prepositioning_rows = velocity_df[velocity_df['step'] == 'prepositioning']

    # Apply the calculate_kinetic_energy function to each row and create a new column for the current link
    kinetic_energy_df[f'kinetic_energy_link_{i}'] = prepositioning_rows.apply(calculate_kinetic_energy, axis=1)

# Include the time column in the DataFrame
kinetic_energy_df['time'] = prepositioning_rows['time']

# Reorder columns to have time as the first column
kinetic_energy_df = kinetic_energy_df[['time'] + [f'kinetic_energy_link_{i}' for i in range(1, 7)]]

# Save the data to a CSV file
kinetic_energy_df.to_csv(WORKDIR / 'kinetic_energy_data_from_graph.csv', index=False)