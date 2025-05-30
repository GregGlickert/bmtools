# Spike Plotting

The `bmplot.spikes` module provides functions for creating various visualizations of neural spike data, including raster plots, population firing rate bar charts, and distributions of individual cell firing rates. These plots are essential for understanding neural activity patterns.

## Raster Plots

Raster plots visualize the spike times of individual neurons over a period. Different populations can be color-coded for clarity.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bmtool.bmplot.spikes import raster

# --- Setup: Generate example spike_df data (replace with your actual data loading) ---
# This would typically come from bmtool.analysis.spikes.load_spikes_to_df()
np.random.seed(123)
num_spikes = 1000
spike_times_data = np.sort(np.random.uniform(0, 2000, num_spikes)) # Spike times in ms over 2 seconds
node_ids_data = np.random.randint(0, 100, num_spikes) # 100 neurons
pop_names_data = ['PopA'] * 50 + ['PopB'] * 50 # Assign neurons to two populations
node_to_pop_mapping = {i: pop_names_data[i] for i in range(100)}
spike_pop_names = [node_to_pop_mapping[nid % 100] for nid in node_ids_data] # Simple mapping for example

spikes_df_example = pd.DataFrame({
    'timestamps': spike_times_data,
    'node_ids': node_ids_data,
    'pop_name': spike_pop_names # This column is used by groupby
})
# --- End of example data setup ---

# Create a raster plot
fig_raster, ax_raster = plt.subplots(figsize=(12, 7))
raster(
    spikes_df=spikes_df_example,
    groupby='pop_name',     # Column in spikes_df to group cells by for coloring
    tstart=0.0,             # Start time in ms for the plot window
    tstop=2000.0,           # Stop time in ms for the plot window
    ax=ax_raster,
    dot_size=0.5,           # Size of the dots in the raster
    # color_map={'PopA': 'blue', 'PopB': 'red'} # Optional: custom colors
)
ax_raster.set_title("Raster Plot by Population")
# plt.show() # Usually not needed if not saving and in a notebook
```

## Firing Rate Statistics Plots

Visualize pre-computed firing rate statistics, such as mean rates per population.

```python
import pandas as pd
import matplotlib.pyplot as plt
from bmtool.bmplot.spikes import plot_firing_rate_pop_stats

# --- Setup: Generate example pop_stats_df data (replace with actual data) ---
# This would typically come from bmtool.analysis.spikes.compute_firing_rate_stats()
pop_stats_example_df = pd.DataFrame({
    'pop_name': ['PopA', 'PopB', 'PopC'],
    'firing_rate_mean': [15.5, 22.1, 12.3], # Mean firing rate in Hz
    'firing_rate_std': [3.2, 4.5, 2.8]      # Standard deviation in Hz
})
# --- End of example data setup ---

# Plot population firing rate statistics (bar plot)
fig_bar, ax_bar = plt.subplots(figsize=(8, 6))
plot_firing_rate_pop_stats(
    firing_stats=pop_stats_example_df, # DataFrame from compute_firing_rate_stats
    groupby='pop_name',                 # The column that defines the groups/bars
    ax=ax_bar
    # color_map={'PopA': 'green', 'PopB': 'orange', 'PopC': 'purple'} # Optional
)
ax_bar.set_title("Mean Firing Rates by Population")
# plt.show()
```

## Firing Rate Distribution Plots

Visualize the distribution of firing rates for individual cells within populations, using box plots, violin plots, or swarm plots.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bmtool.bmplot.spikes import plot_firing_rate_distribution

# --- Setup: Generate example individual_stats_df data (replace with actual data) ---
# This would also typically come from bmtool.analysis.spikes.compute_firing_rate_stats()
np.random.seed(456)
n_cells_dist = 150
individual_stats_example_df = pd.DataFrame({
    'node_ids': range(n_cells_dist),
    'firing_rate': np.concatenate([
        np.random.normal(10, 3, 50), 
        np.random.normal(20, 5, 50), 
        np.random.normal(15, 4, 50)
    ]), # Firing rates in Hz
    'pop_name': ['Group1'] * 50 + ['Group2'] * 50 + ['Group3'] * 50
})
individual_stats_example_df['firing_rate'] = individual_stats_example_df['firing_rate'].clip(lower=0) # No negative rates
# --- End of example data setup ---

# Plot firing rate distributions (e.g., box plot combined with swarm plot)
fig_dist, ax_dist = plt.subplots(figsize=(10, 7))
plot_firing_rate_distribution(
    individual_stats=individual_stats_example_df, # DataFrame from compute_firing_rate_stats
    groupby='pop_name',
    plot_type=['box', 'swarm'], # Overlay box plot and swarm plot
    ax=ax_dist,
    swarm_alpha=0.5
)
ax_dist.set_title("Individual Firing Rate Distributions by Population")
# plt.show()
```

These plotting functions help in quickly assessing spike train characteristics and population dynamics from simulation outputs.
