# Spike Analysis

This module provides functions for loading, analyzing, and calculating statistics from neural spike data from BMTK simulations. This includes calculating firing rates for individual cells and populations, handling time windows, and identifying specific firing patterns like bursting or high-frequency firing cells.

## Loading and Processing Spike Data

The first step in spike analysis is often loading the spike times and associated node information.

```python
import pandas as pd
from bmtool.analysis.spikes import load_spikes_to_df, compute_firing_rate_stats

# Load spike data from a simulation
# Ensure 'output/spikes.h5' and 'config.json' are correct paths for your setup.
# The 'network_name' should match the network name in your simulation files.
spikes_df = load_spikes_to_df(
    spike_file='output/spikes.h5',
    network_name='network', # Replace 'network' with your actual network name if different
    config_path='config.json',  # Optional: Path to simulation config for population info
    groupby='pop_name'          # Column from node properties to group by (e.g., 'pop_name')
)

# Display some basic information about the loaded spikes
print("Spikes DataFrame head:")
print(spikes_df.head())
print(f"\nTotal spikes loaded: {len(spikes_df)}")

# Get basic spike statistics by population over a specific time window
# t_start and t_stop are in milliseconds.
pop_stats_df, individual_stats_df = compute_firing_rate_stats(
    df=spikes_df,
    groupby='pop_name',
    t_start=500.0,  # e.g., start analysis at 500 ms
    t_stop=1500.0   # e.g., end analysis at 1500 ms
)

print("\nPopulation firing rate statistics (mean and std):")
print(pop_stats_df)

print("\nIndividual cell firing rate statistics (first 5):")
print(individual_stats_df.head())
```

## Population Spike Rates

Calculate and visualize population-level spike rates over time. This can reveal overall activity patterns in different cell groups.

```python
import matplotlib.pyplot as plt
from bmtool.analysis.spikes import get_population_spike_rate

# Calculate population spike rates over time
# fs: sampling frequency for the rate calculation (e.g., how many bins per second)
# t_stop: can be inferred if not provided, but explicit is good.
# Ensure 'config.json' and 'network' name are correct if you want accurate per-neuron rates.
population_rates_xarray = get_population_spike_rate(
    spike_data=spikes_df, # DataFrame from load_spikes_to_df
    fs=100.0,             # Effective sampling frequency for rate (e.g., 10 ms bins)
    t_start=0.0,          # Start time in ms
    t_stop=2000.0,        # Stop time in ms
    config_path='config.json', 
    network_name='network',
    normalize=False,      # Set to True to normalize rates to [0,1] per population
    smooth_window=5       # Smoothing window in number of time bins (e.g., 5 bins * 10ms/bin = 50ms smoothing)
)

# Plot population rates (raw and smoothed are available in the xarray)
plt.figure(figsize=(12, 6))
for pop_name in population_rates_xarray.population.data:
    # Plot smoothed rates
    plt.plot(population_rates_xarray.time.data, 
             population_rates_xarray.sel(population=pop_name, type='smoothed').data, 
             label=f"{pop_name} (smoothed)")
plt.xlabel('Time (ms)')
plt.ylabel('Population Firing Rate (Hz/neuron)')
plt.legend()
plt.title('Population Firing Rates (Smoothed)')
plt.show()
```

## Advanced Spike Analysis

Identifying bursting cells or cells with the highest firing rates.

```python
from bmtool.analysis.spikes import (
    find_bursting_cells, 
    find_highest_firing_cells,
    average_spike_rate_over_windows,
    compare_firing_over_times
)

# Find bursting cells (example parameters)
# This function modifies 'pop_name' for bursting cells by appending '_bursters'.
bursting_cells_df = find_bursting_cells(
    df=spikes_df.copy(), # Use a copy if you don't want to modify the original spikes_df
    isi_threshold=10.0,  # Max ISI in ms to be considered part of a burst
    burst_count_threshold=2 # Min number of short ISIs to be a burster
)
print("\nUnique population names after identifying bursters:")
print(bursting_cells_df['pop_name'].unique())

# Find the top 20% highest firing cells in each population
# (upper_quantile=0.8 means cells above the 80th percentile)
highest_firing_df = find_highest_firing_cells(
    df=spikes_df, 
    upper_quantile=0.8, 
    groupby='pop_name'
)
print(f"\nNumber of spikes from top 20% high-firing cells: {len(highest_firing_df)}")


# Example for averaging spike rate over windows (using previously calculated population_rates_xarray)
# Define some example time windows in ms
example_windows = [(500.0, 700.0), (1000.0, 1200.0)] 
avg_rate_over_windows = average_spike_rate_over_windows(
    spike_rate=population_rates_xarray, # This is an xarray.DataArray
    windows=example_windows
)
print("\nAverage population rates over specified windows:")
print(avg_rate_over_windows)


# Example for comparing firing rates between two time windows for each population
print("\nComparison of firing rates between two time windows:")
compare_firing_over_times(
    spike_df=spikes_df,
    group_by='pop_name',
    time_window_1=(500.0, 1000.0), # First window (start_ms, stop_ms)
    time_window_2=(1000.0, 1500.0) # Second window
)
```

## Visualizing Spike Data

Use the `bmtool.bmplot.spikes` module for dedicated spike data visualizations.

```python
from bmtool.bmplot.spikes import raster, plot_firing_rate_pop_stats, plot_firing_rate_distribution
# Assuming spikes_df, pop_stats_df, and individual_stats_df are available from previous examples.

# Create a raster plot
# Note: The 'raster' function in bmplot.spikes might have different parameter names or capabilities.
# This example assumes it's compatible with the refactored version.
# Check bmplot.spikes.raster docstring for current parameters.
fig_raster, ax_raster = plt.subplots(figsize=(12, 7))
raster(
    spikes_df=spikes_df,
    groupby='pop_name', # Column to group for colors
    tstart=0.0,         # Start time in ms
    tstop=2000.0,       # Stop time in ms
    ax=ax_raster,
    dot_size=0.5
)
ax_raster.set_title("Raster Plot by Population")
plt.show()

# Plot population firing rate statistics (bar plot)
fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
plot_firing_rate_pop_stats(
    firing_stats=pop_stats_df, # DataFrame from compute_firing_rate_stats
    groupby='pop_name',
    ax=ax_bar
)
plt.show()

# Plot firing rate distributions (e.g., box plot)
fig_dist, ax_dist = plt.subplots(figsize=(10, 6))
plot_firing_rate_distribution(
    individual_stats=individual_stats_df, # DataFrame from compute_firing_rate_stats
    groupby='pop_name',
    plot_type='box', # Can be 'box', 'violin', 'swarm', or a list like ['box', 'swarm']
    ax=ax_dist
)
plt.show()
```
This provides a foundation for analyzing spike data. Remember to adapt paths and parameters to your specific simulation setup.
