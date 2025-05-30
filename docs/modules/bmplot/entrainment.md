# Entrainment Plotting

The `bmtool.bmplot.entrainment` module provides functions for visualizing neural entrainment phenomena, such as the relationship between spike timing and LFP oscillations (phase-locking, spike-field coherence) and other population-level entrainment statistics.

## Spike Phase Distribution relative to LFP Cycle

Visualize how spike times are distributed across phases of an LFP oscillation.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bmtool.bmplot.entrainment import plot_cycle_with_spike_histograms
from bmtool.analysis.lfp import get_lfp_phase # For generating example phase_data
from bmtool.analysis.spikes import load_spikes_to_df # For example spike data

# --- Setup: Generate example data (replace with your actual data loading) ---
# Example LFP data (replace with actual LFP loading and filtering)
fs = 1000.0  # Sampling rate of LFP
lfp_duration_s = 2
t = np.linspace(0, lfp_duration_s, int(fs * lfp_duration_s), endpoint=False)
# Create a dummy LFP signal (e.g., a sine wave for a specific band like theta)
# In a real scenario, you would load and filter your LFP data.
# Here, we use a simple sine wave for demonstration.
lfp_signal_dummy = np.sin(2 * np.pi * 6 * t) # 6 Hz theta example

# Example spike data (replace with actual spike loading)
# Create a dummy spikes_df DataFrame
np.random.seed(0)
n_spikes_pop1 = 200
n_spikes_pop2 = 150
spike_times_pop1 = np.sort(np.random.uniform(0, lfp_duration_s * 1000, n_spikes_pop1)) # ms
spike_times_pop2 = np.sort(np.random.uniform(0, lfp_duration_s * 1000, n_spikes_pop2)) # ms

spikes_df_example = pd.DataFrame({
    'timestamps': np.concatenate([spike_times_pop1, spike_times_pop2]),
    'node_ids': np.concatenate([np.arange(n_spikes_pop1), np.arange(n_spikes_pop2)]), # Dummy node_ids
    'pop_name': ['PopA'] * n_spikes_pop1 + ['PopB'] * n_spikes_pop2
})

# Calculate LFP phase (example for theta band 4-8 Hz)
# In a real case, ensure lfp_signal_dummy is your actual filtered LFP or pass raw LFP to get_lfp_phase
# For this example, we'll generate phases directly from our dummy LFP.
# If lfp_signal_dummy was raw, you'd use:
# lfp_phase_rad = get_lfp_phase(lfp_data=lfp_signal_dummy, fs=fs, filter_method='butter', lowcut=4, highcut=8)
# Here, since it's already a sine wave, its phase is known (or can be obtained via Hilbert)
from scipy.signal import hilbert
lfp_phase_rad = np.angle(hilbert(lfp_signal_dummy))


# Simulate extracting spike phases for each population
phase_data_example = {}
for pop_name in spikes_df_example['pop_name'].unique():
    pop_spike_times_ms = spikes_df_example[spikes_df_example['pop_name'] == pop_name]['timestamps'].values
    # Convert spike times from ms to seconds to match LFP time 't'
    pop_spike_times_s = pop_spike_times_ms / 1000.0
    # Find nearest LFP phase for each spike time
    spike_indices = np.searchsorted(t, pop_spike_times_s, side="left")
    # Ensure indices are within bounds
    spike_indices = np.clip(spike_indices, 0, len(lfp_phase_rad) - 1)
    phase_data_example[pop_name] = lfp_phase_rad[spike_indices]
# --- End of example data setup ---

# Plot spike phase distribution relative to an idealized cycle
plot_cycle_with_spike_histograms(
    phase_data=phase_data_example,
    bins=18,  # Bins for the histogram (e.g., 18 bins for 20-degree bins)
    pop_names_to_plot=['PopA', 'PopB'] # Specify which populations to plot
    # save_path="cycle_spike_hist.png" # Optional
)
# plt.show() is usually handled by the function if not saving and not in notebook
```

## Population Entrainment Metrics Across Frequencies

Visualize how entrainment metrics (like PPC or PLV) for different populations vary across a range of frequencies.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bmtool.bmplot.entrainment import plot_entrainment_by_population

# --- Setup: Generate example entrainment_data (replace with your actual data) ---
# This dictionary should come from bmtool.analysis.entrainment.calculate_entrainment_per_cell
pop_names_example = ['Excitatory', 'Inhibitory']
freqs_example = np.array([10, 20, 30, 40, 50, 60, 70, 80])
entrainment_data_example = {
    pop: {
        node_id: {
            freq: np.random.rand() * (0.1 + idx_pop * 0.2) + (freq/100)*0.1 # Synthetic data
            for freq in freqs_example
        }
        for node_id in range(50) # 50 cells per population
    }
    for idx_pop, pop in enumerate(pop_names_example)
}
# --- End of example data setup ---

# Plot entrainment by frequency for specified populations
plot_entrainment_by_population(
    ppc_dict=entrainment_data_example, # Pass the entrainment data here
    pop_names=pop_names_example,
    freqs=freqs_example,
    title="Population Entrainment (e.g., PPC) by Frequency",
    y_label="Mean PPC Value", # Adjust if using PLV or other metric
    # save_path="entrainment_by_freq.png" # Optional
)
```

## Distribution of Entrainment Values (Swarm Plot)

Visualize the distribution of entrainment values for individual cells within populations at a specific frequency.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bmtool.bmplot.entrainment import plot_entrainment_swarm_plot

# Assuming entrainment_data_example and pop_names_example from previous example
specific_frequency = 40.0 # Hz

plot_entrainment_swarm_plot(
    entrainment_data=entrainment_data_example,
    pop_names=pop_names_example,
    freq=specific_frequency,
    title=f"Cellular Entrainment at {specific_frequency} Hz",
    y_label="PPC Value", # Adjust if using PLV etc.
    # save_path=f"entrainment_swarm_{specific_frequency}Hz.png" # Optional
)
```

## Trial-Averaged Entrainment

Plot how entrainment metrics average across multiple trials or time windows.

```python
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from bmtool.bmplot.entrainment import plot_trial_avg_entrainment

# --- Setup: Generate example data (replace with your actual data loading) ---
# Example spike_df (timestamps in ms)
np.random.seed(42)
num_trials = 5
trial_duration_ms = 1000
total_duration_ms = num_trials * trial_duration_ms
pop_names_trial_example = ['PopX', 'PopY']
spike_data_list = []
for trial in range(num_trials):
    for pop_idx, pop_name in enumerate(pop_names_trial_example):
        num_spikes_trial_pop = np.random.randint(50, 150)
        spike_times_trial_pop = np.sort(np.random.uniform(0, trial_duration_ms, num_spikes_trial_pop)) + trial * trial_duration_ms
        for spike_time in spike_times_trial_pop:
            spike_data_list.append({'timestamps': spike_time, 'node_ids': np.random.randint(0, 20) + pop_idx*20, 'pop_name': pop_name})
spike_df_trial_example = pd.DataFrame(spike_data_list)

# Example LFP data (as xr.DataArray)
lfp_fs_example = 1000.0 # Hz
time_coords = np.arange(0, total_duration_ms, 1000.0/lfp_fs_example)
lfp_signal_trial_example = np.random.randn(len(time_coords)) # Replace with actual LFP
lfp_xr_example = xr.DataArray(lfp_signal_trial_example, coords={'time': time_coords}, dims=['time'], attrs={'fs': lfp_fs_example})

# Example time windows (ms)
time_windows_example = [(i * trial_duration_ms, (i + 1) * trial_duration_ms) for i in range(num_trials)]

# Frequencies to analyze
freqs_trial_example = np.linspace(10, 100, 10)
# --- End of example data setup ---

plot_trial_avg_entrainment(
    spike_df=spike_df_trial_example,
    lfp_data=lfp_xr_example, # Corrected from lfp to lfp_data
    time_windows=time_windows_example,
    entrainment_method='ppc', # 'ppc', 'ppc2', or 'plv'
    pop_names=pop_names_trial_example,
    freqs=freqs_trial_example,
    firing_quantile=0.5, # Consider top 50% of firing cells in each trial window for this example
    spike_fs=1000.0,     # Assuming spike times are in ms, so 1000 Hz effective sampling for conversion
    lfp_fs=_lfp_fs_example,      # Pass LFP sampling rate
    error_type='ci'      # 'ci', 'sem', or 'std'
    # save_path="trial_avg_entrainment.png" # Optional
)
```
