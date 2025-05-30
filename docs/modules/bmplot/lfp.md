# LFP/ECP Plotting

The `bmtool.bmplot.lfp` module currently provides functions for visualizing LFP/ECP data, primarily focusing on spectrograms. Power spectra and time-series plots can be generated using standard Matplotlib and SciPy functions, often in conjunction with analysis results from `bmtool.analysis.lfp`.

## Spectrograms

Visualizing the time-frequency representation of LFP data is crucial for identifying oscillations and their dynamics.

```python
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from bmtool.analysis.lfp import cwt_spectrogram_xarray # For generating example data
from bmtool.bmplot.lfp import plot_spectrogram

# --- Setup: Generate example spectrogram data (replace with your actual data) ---
fs_example = 1000.0  # Sampling rate in Hz
time_example = np.linspace(0, 5, int(fs_example * 5), endpoint=False) # 5 seconds of data
# Create a dummy signal: 10 Hz alpha + 50 Hz gamma burst
signal_alpha = np.sin(2 * np.pi * 10 * time_example)
gamma_burst_time = (time_example > 2) & (time_example < 3)
signal_gamma = np.sin(2 * np.pi * 50 * time_example) * gamma_burst_time * 2
dummy_lfp_data = signal_alpha + signal_gamma + np.random.randn(len(time_example)) * 0.5

# Calculate spectrogram using a function from the analysis module
# This returns an xarray.Dataset with 'PSD' and 'cone_of_influence_frequency'
spectrogram_dataset = cwt_spectrogram_xarray(
    x=dummy_lfp_data,
    fs=fs_example,
    freq_range=(1, 100), # Frequencies from 1 to 100 Hz
    nNotes=12, # Number of voices per octave for CWT
    bandwidth=1.0 
)
# --- End of example data setup ---

# Plot the spectrogram
fig, ax = plt.subplots(figsize=(12, 6))
plot_spectrogram(
    psd_xarray=spectrogram_dataset, # Pass the Dataset; function expects PSD DataArray with coords
    log_power=True,              # Log-scale the power
    plt_range=(1, 100),          # Frequency range to display on the y-axis
    color_map_freq_range=(5, 70),# Freq range to determine color limits
    ax=ax
    # save_path="spectrogram.png" # Optional
)
ax.set_title("LFP Spectrogram Example")
# plt.show() # Usually not needed if not saving and in a notebook
```

## Plotting Power Spectra (using analysis results)

While `bmplot.lfp` doesn't have a dedicated power spectrum plotting function currently, you can easily plot results from `bmtool.analysis.lfp.fit_fooof` or `scipy.signal.welch`:

```python
from bmtool.analysis.lfp import fit_fooof # Assuming FOOOF is installed
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np # Ensure numpy is imported

# --- Setup: Assuming dummy_lfp_data and fs_example from above ---
# Or load your LFP data, e.g., as a 1D numpy array
# lfp_signal = my_lfp_data_channel_x.data 
# fs = my_lfp_data_channel_x.attrs['fs']

# Calculate PSD using Welch's method
freqs, pxx = signal.welch(dummy_lfp_data, fs=fs_example, nperseg=min(1024, len(dummy_lfp_data)))

# Fit FOOOF model (optional, for plotting fit)
try:
    fooof_results, fooof_model_obj = fit_fooof(
        f=freqs,
        pxx=pxx,
        freq_range=(1.0, 100.0),
        peak_width_limits=(1.0, 12.0) # Adjusted example limits
    )
except ImportError: # Handle if fooof is not installed
    fooof_model_obj = None
    print("FOOOF not installed, skipping FOOOF model plot.")
# --- End of example data setup ---

# Plot power spectrum
plt.figure(figsize=(10, 6))
plt.plot(freqs, pxx, label="PSD (Welch)")
if fooof_model_obj:
    fooof_model_obj.plot(ax=plt.gca(), plt_log=False, add_legend=False) # Plot FOOOF fit on same axes
    plt.legend(["PSD (Welch)", "FOOOF Full Fit", "FOOOF Aperiodic Fit"]) # Manual legend
else:
    plt.legend()
    
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectral Density") # Adjust units if necessary e.g. V^2/Hz or dB
plt.title("LFP Power Spectrum")
# plt.loglog() # Often useful for PSDs
plt.grid(True, alpha=0.5)
plt.show()
```

## Plotting LFP Time Series

For plotting raw LFP time series, standard Matplotlib functions can be used directly on your `xr.DataArray` or `np.ndarray` LFP data.

```python
import matplotlib.pyplot as plt
import xarray as xr # Assuming LFP data might be in an xarray DataArray

# --- Setup: Assuming lfp_data (xr.DataArray) from the first example ---
# Or create/load your LFP data:
# time_coords = np.arange(0, 1000, 1) # 1000 ms at 1kHz
# lfp_example_ts = xr.DataArray(np.random.randn(1000), coords={'time': time_coords}, dims=['time'], name='LFP_channel_0')
# lfp_example_ts.attrs['fs'] = 1000.0
# --- End of example data setup ---

# Select a channel if multi-channel, and a time range
# lfp_to_plot = lfp_data.sel(channel_id=lfp_data.channel_id.data[0], time=slice(0, 500)) # Example slice

# For this example, using the dummy_lfp_data and time_example from spectrogram section
lfp_to_plot_data = dummy_lfp_data
lfp_to_plot_time = time_example * 1000 # Convert to ms for consistency if preferred

plt.figure(figsize=(12, 4))
plt.plot(lfp_to_plot_time[:2000], lfp_to_plot_data[:2000]) # Plot first 2000 ms
plt.xlabel("Time (ms)")
plt.ylabel("LFP Amplitude (uV)") # Adjust unit as appropriate
plt.title("LFP Time Series (Example Channel)")
plt.grid(True, alpha=0.5)
plt.show()
```
