import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
from scipy.ndimage import gaussian_filter1d

# Define the working directory for the specific data
WORKDIR = Path(r'C:\Users\Chirag\Desktop\fhof\load-layers\load-layers_horizontal_close_vel-0.25_rep-1\processed')

# Step 1: Import the CSV file containing velocity data
velocity_df = pd.read_csv(WORKDIR / 'sensordata_processed.csv', low_memory=False)

# Step 2: Import the CSV file containing inertia data for multiple robot links
inertial_df = pd.read_csv(WORKDIR / 'parameters_inertial.csv', index_col="body_name", header=0)

# Get the list of link IDs
link_ids = inertial_df.index.to_list()

# Define the steps of interest
steps_of_interest = ['travel_1', 'travel_2']

# Initialize a dictionary to store total kinetic energy for each step
total_kinetic_energy_dict = {}

# Loop through the steps of interest
for step in steps_of_interest:
    # Filter rows based on the current step
    step_rows = velocity_df[velocity_df['step'] == step]

    # Initialize a column for total kinetic energy in the step DataFrame
    step_rows['total_kinetic_energy'] = 0

    # Loop through the first six links
    for i in range(1, 7):
        # Extract the required columns for linear and angular velocities for the current link
        link_id = link_ids[i - 1]  # Adjust the index to start from 0
        linvel_columns = [f'robot_{link_id}_linvel.{axis}' for axis in "xyz"]
        angvel_columns = [f'robot_{link_id}_angvel.{axis}' for axis in "xyz"]

        # Check if the columns exist in the filtered DataFrame
        if not set(linvel_columns).issubset(step_rows.columns) or not set(angvel_columns).issubset(step_rows.columns):
            continue

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
        step_rows[f'kinetic_energy_link_{i}'] = step_rows.apply(calculate_kinetic_energy, axis=1)

        # Sum the kinetic energy of the current link to the total kinetic energy
        step_rows['total_kinetic_energy'] += step_rows[f'kinetic_energy_link_{i}']

    # Calculate the total kinetic energy for this step
    absolute_power = step_rows['total_kinetic_energy'].diff(1).abs()
    total_kinetic_energy = absolute_power.sum() * step_rows["time"].mean()

    # Store the total kinetic energy for the current step
    total_kinetic_energy_dict[step] = total_kinetic_energy

    # Print the total kinetic energy for the current step
    print(f"Total Kinetic Energy ({step}):", total_kinetic_energy)

    # Smooth the total kinetic energy data using a Gaussian filter with a moderate sigma value
    sigma_value = 3  # Moderate sigma value for smoother curve without over-smoothing
    step_rows['smoothed_kinetic_energy'] = gaussian_filter1d(step_rows['total_kinetic_energy'], sigma=sigma_value)

    # Plot the smoothed total kinetic energy over time for this step
    fig = px.line(step_rows, x='time', y='smoothed_kinetic_energy', title=f'Smoothed Total Kinetic Energy over Time for {step}', labels={'smoothed_kinetic_energy': 'Smoothed Total Kinetic Energy', 'time': 'Time'})
    fig.show()

# Optionally, print the total kinetic energy values
print("Total Kinetic Energy for all steps:", total_kinetic_energy_dict)