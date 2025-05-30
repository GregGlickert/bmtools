# LFP/ECP Analysis

This module provides tools for analyzing Local Field Potentials (LFP) and Extracellular Potentials (ECP) from BMTK simulations, including loading, filtering, spectral analysis, and feature extraction.

## Loading and Processing LFP/ECP Data

```python
import numpy as np
import xarray as xr
from bmtool.analysis.lfp import load_ecp_to_xarray, ecp_to_lfp, slice_time_series

# Load ECP data
# Assuming ecp.h5 is in an 'output' directory relative to where the script/notebook is run
ecp_data = load_ecp_to_xarray('output/ecp.h5', demean=True)

# Convert ECP to LFP with filtering
# fs should match the sampling frequency of ecp_data, which is often an attribute of ecp_data
lfp_data = ecp_to_lfp(
    ecp_data=ecp_data,
    cutoff=250,        # Cutoff frequency in Hz
    fs=ecp_data.attrs.get('fs', 10000.0) # Get fs from xarray attributes or default
)

# Slice data to specific time range (e.g., 500ms to 1500ms)
# Ensure time coordinates in lfp_data are in milliseconds if start/stop are in ms
lfp_slice = slice_time_series(lfp_data, time_ranges=(500, 1500))
```

## Spectral Analysis

Analyze frequency content using wavelets and FOOOF.

```python
from bmtool.analysis.lfp import (
    cwt_spectrogram_xarray,
    fit_fooof,
    generate_resd_from_fooof
)
import matplotlib.pyplot as plt
from scipy import signal

# Assuming lfp_slice and lfp_data are available from the previous example
# And that lfp_data has a single channel or you select one:
lfp_channel_data = lfp_slice.sel(channel_id=lfp_slice.channel_id.data[0]).data
lfp_sampling_frequency = lfp_slice.attrs.get('fs', 1000.0) # Use downsampled fs if available

# Calculate wavelet spectrogram
# Note: cwt_spectrogram_xarray expects a numpy array for x
spectrogram_dataset = cwt_spectrogram_xarray(
    x=lfp_channel_data,
    fs=lfp_sampling_frequency, # Use the correct sampling frequency for the data slice
    freq_range=(1, 100), # Example frequency range
    nNotes=8 # Example number of notes for wavelet scales
)

# Calculate power spectrum for FOOOF using data from the full LFP (or relevant slice)
# For Welch's method, ensure data is 1D
lfp_full_channel_data = lfp_data.sel(channel_id=lfp_data.channel_id.data[0]).data
fs_full = lfp_data.attrs.get('fs', 10000.0) # Original sampling rate if lfp_data is from ecp_to_lfp without downsampling

freqs, pxx = signal.welch(lfp_full_channel_data, fs=fs_full, nperseg=min(4096, len(lfp_full_channel_data)))

# Fit FOOOF model
# The fit_fooof function returns: results, fooof_model_object
fooof_results, fooof_model_obj = fit_fooof(
    f=freqs,
    pxx=pxx,
    freq_range=(1.0, 100.0), # Must be within the range of freqs from Welch
    peak_width_limits=(1.0, 8.0),
    max_n_peaks=6
)
# fooof_model_obj.plot() # To plot the FOOOF model fit
# plt.show() # If not in a notebook

# Get residuals between original spectrum and aperiodic fit
# generate_resd_from_fooof expects the FOOOF model object
res_psd, ap_fit = generate_resd_from_fooof(fooof_model_obj)
```

## Filtering and LFP Features

Apply various filters and extract features like power and phase.

```python
from bmtool.analysis.lfp import (
    butter_bandpass_filter, 
    wavelet_filter, 
    calculate_SNR, 
    get_lfp_power, 
    get_lfp_phase,
    calculate_wavelet_passband
)

# Assuming lfp_data and fooof_model_obj are available from previous examples
lfp_channel_data = lfp_data.sel(channel_id=lfp_data.channel_id.data[0]).data
current_fs = lfp_data.attrs.get('fs', 10000.0)

# Band-pass filter
gamma_signal_butter = butter_bandpass_filter(
    data=lfp_channel_data,
    lowcut=30, # Hz
    highcut=80, # Hz
    fs=current_fs
)

# Wavelet filter centered at specific frequency
center_freq_gamma = 40.0 # Hz
bandwidth_gamma = 10.0 # Hz (parameter for Morlet wavelet)
gamma_signal_wavelet = wavelet_filter(
    x=lfp_channel_data,
    freq=center_freq_gamma,
    fs=current_fs,
    bandwidth=bandwidth_gamma 
)
# You can also check the effective passband of the wavelet
lower_bound, upper_bound, passband_width = calculate_wavelet_passband(center_freq_gamma, bandwidth_gamma)
print(f"Wavelet at {center_freq_gamma} Hz with bandwidth param {bandwidth_gamma}: Passband [{lower_bound:.2f} - {upper_bound:.2f}] Hz, Width: {passband_width:.2f} Hz")


# Calculate signal-to-noise ratio using a fitted FOOOF model
# Ensure fooof_model_obj is fitted to the spectrum of the data you are analyzing
snr_gamma = calculate_SNR(fooof_model_obj, freq_band=(30.0, 80.0))
print(f"Signal-to-noise ratio in gamma band (30-80 Hz): {snr_gamma:.2f}")

# Get LFP power in a specific band using wavelet
lfp_power_gamma_wavelet = get_lfp_power(
    lfp_data=lfp_data.sel(channel_id=lfp_data.channel_id.data[0]), # Can pass DataArray
    freq=40.0, # Center frequency
    fs=current_fs, # Must match actual sampling frequency of lfp_data
    filter_method='wavelet',
    bandwidth=10.0
)
print(f"Mean Gamma power (wavelet): {lfp_power_gamma_wavelet.mean().item()}")

# Get LFP power in a specific band using Butterworth filter
lfp_power_theta_butter = get_lfp_power(
    lfp_data=lfp_data.sel(channel_id=lfp_data.channel_id.data[0]).data, # Can pass numpy array
    fs=current_fs,
    filter_method='butter',
    lowcut=4.0,
    highcut=8.0
)
print(f"Mean Theta power (Butterworth): {np.mean(lfp_power_theta_butter)}")


# Get LFP phase at a specific frequency using wavelet
lfp_phase_theta_wavelet = get_lfp_phase(
    lfp_data=lfp_data.sel(channel_id=lfp_data.channel_id.data[0]),
    freq=6.0, # Center frequency for phase
    fs=current_fs,
    filter_method='wavelet',
    bandwidth=2.0 # Typically narrower for phase
)
print(f"Example Theta phases (first 5 points): {lfp_phase_theta_wavelet.data[:5]}")

# Get LFP phase in a band using Butterworth + Hilbert
lfp_phase_gamma_butter = get_lfp_phase(
    lfp_data=lfp_data.sel(channel_id=lfp_data.channel_id.data[0]).data,
    fs=current_fs,
    filter_method='butter',
    lowcut=30.0,
    highcut=80.0
)
print(f"Example Gamma phases (Butterworth, first 5 points): {lfp_phase_gamma_butter[:5]}")
```

This provides a more comprehensive overview of LFP analysis capabilities.
