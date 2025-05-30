"""
Module for processing BMTK LFP output.
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import xarray as xr
from fooof import FOOOF
from fooof.sim.gen import gen_model
from scipy import signal

from ..bmplot.connections import is_notebook


def load_ecp_to_xarray(ecp_file: str, demean: bool = False) -> xr.DataArray:
    """
    Load ECP data from an HDF5 file (BMTK sim) into an xarray DataArray.

    Parameters
    ----------
    ecp_file : str
        Path to the HDF5 file containing ECP data.
    demean : bool, optional
        If True, the mean of the data will be subtracted (default is False).

    Returns
    -------
    xr.DataArray
        An xarray DataArray containing the ECP data, with time as one dimension
        and channel_id as another.
    """
    with h5py.File(ecp_file, "r") as f:
        ecp = xr.DataArray(
            f["ecp"]["data"][()].T,
            coords=dict(
                channel_id=f["ecp"]["channel_id"][()],
                time=np.arange(*f["ecp"]["time"]),  # ms
            ),
            attrs=dict(
                fs=1000 / f["ecp"]["time"][2]  # Hz
            ),
        )
    if demean:
        ecp -= ecp.mean(dim="time")
    return ecp


def ecp_to_lfp(
    ecp_data: xr.DataArray, cutoff: float = 250, fs: float = 10000, downsample_freq: float = 1000
) -> xr.DataArray:
    """
    Apply a low-pass Butterworth filter to an xarray DataArray and optionally downsample.
    This filters out the high end frequencies turning the ECP into a LFP

    Parameters
    ----------
    ecp_data : xr.DataArray
        The input data array containing LFP data with time as one dimension.
    cutoff : float, optional
        The cutoff frequency for the low-pass filter in Hz (default is 250Hz).
    fs : float, optional
        The sampling frequency of the data (default is 10000 Hz).
    downsample_freq : Optional[float], optional
        The frequency to downsample to. If None, no downsampling is performed (default is 1000 Hz).

    Returns
    -------
    xr.DataArray
        The filtered (and possibly downsampled) data as an xarray DataArray.
    """
    # Bandpass filter design
    nyq = 0.5 * fs
    cut = cutoff / nyq
    b, a = signal.butter(8, cut, btype="low", analog=False)

    # Initialize an array to hold filtered data
    filtered_data = xr.DataArray(
        np.zeros_like(ecp_data), coords=ecp_data.coords, dims=ecp_data.dims
    )

    # Apply the filter to each channel
    for channel in ecp_data.channel_id:
        filtered_data.loc[channel, :] = signal.filtfilt(
            b, a, ecp_data.sel(channel_id=channel).values
        )

    # Downsample the filtered data if a downsample frequency is provided
    if downsample_freq is not None:
        downsample_factor = int(fs / downsample_freq)
        filtered_data = filtered_data.isel(time=slice(None, None, downsample_factor))
        # Update the sampling frequency attribute
        filtered_data.attrs["fs"] = downsample_freq

    return filtered_data


from typing import List, Tuple, Union, Optional, Dict # Added Dict
# ... (other imports)

def slice_time_series(data: xr.DataArray, time_ranges: Union[Tuple[float, float], List[Tuple[float, float]]]) -> xr.DataArray:
    """
    Slice the xarray DataArray based on provided time ranges.
    Can be used to get LFP during certain stimulus times.

    Parameters
    ----------
    data : xr.DataArray
        The input xarray DataArray containing time-series data.
    time_ranges : Union[Tuple[float, float], List[Tuple[float, float]]]
        One or more tuples representing the (start, stop) time points for slicing.
        For example: (start, stop) or [(start1, stop1), (start2, stop2)].

    Returns
    -------
    xr.DataArray
        A new xarray DataArray containing the concatenated slices.
    """
    # Ensure time_ranges is a list of tuples
    if isinstance(time_ranges, tuple) and len(time_ranges) == 2:
        time_ranges = [time_ranges]

    # List to hold sliced data
    slices = []

    # Slice the data for each time range
    for start, stop in time_ranges:
        sliced_data = data.sel(time=slice(start, stop))
        slices.append(sliced_data)

    # Concatenate all slices along the time dimension if more than one slice
    if len(slices) > 1:
        return xr.concat(slices, dim="time")
    else:
        return slices[0]


def fit_fooof(
    f: np.ndarray,
    pxx: np.ndarray,
    aperiodic_mode: str = "fixed",
    dB_threshold: float = 3.0,
    max_n_peaks: int = 10,
    freq_range: Optional[Tuple[float, float]] = None,
    peak_width_limits: Optional[Tuple[float, float]] = None,
    report: bool = False,
    plot: bool = False,
    plt_log: bool = False,
    plt_range: Optional[Tuple[float, float]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
) -> Tuple[Dict, FOOOF]:
    """
    Fit a FOOOF model to power spectral density data.

    Parameters
    ----------
    f : np.ndarray
        Frequencies corresponding to the power spectral density data.
    pxx : np.ndarray
        Power spectral density data to fit.
    aperiodic_mode : str, optional
        The mode for fitting aperiodic components ('fixed' or 'knee', default is 'fixed').
    dB_threshold : float, optional
        Minimum peak height in dB (default is 3.0).
    max_n_peaks : int, optional
        Maximum number of peaks to fit (default is 10).
    freq_range : Optional[Tuple[float, float]], optional
        Frequency range to fit (default is None, which uses the full range).
    peak_width_limits : Optional[Tuple[float, float]], optional
        Limits on the width of peaks (default is None).
    report : bool, optional
        If True, will print fitting results (default is False).
    plot : bool, optional
        If True, will plot the fitting results (default is False).
    plt_log : bool, optional
        If True, use a logarithmic scale for the y-axis in plots (default is False).
    plt_range : Optional[Tuple[float, float]], optional
        Range for plotting (default is None).
    figsize : Optional[Tuple[float, float]], optional
        Size of the figure (default is None).
    title : Optional[str], optional
        Title for the plot (default is None).

    Returns
    -------
    Tuple[Dict, FOOOF]
        A tuple containing the fitting results (dict-like FOOOFResult object) and the FOOOF model object.
    """
    if aperiodic_mode != "knee":
        aperiodic_mode = "fixed"

    def set_range(x, upper=f[-1]):
        x = np.array(upper) if x is None else np.array(x)
        return [f[2], x.item()] if x.size == 1 else x.tolist()

    freq_range = set_range(freq_range)
    peak_width_limits = set_range(peak_width_limits, np.inf)

    # Initialize a FOOOF object
    fm = FOOOF(
        peak_width_limits=peak_width_limits,
        min_peak_height=dB_threshold / 10,
        peak_threshold=0.0,
        max_n_peaks=max_n_peaks,
        aperiodic_mode=aperiodic_mode,
    )

    # Fit the model
    try:
        fm.fit(f, pxx, freq_range)
    except Exception as e:
        fl = np.linspace(f[0], f[-1], int((f[-1] - f[0]) / np.min(np.diff(f))) + 1)
        fm.fit(fl, np.interp(fl, f, pxx), freq_range)

    results = fm.get_results()

    if report:
        fm.print_results()
        if aperiodic_mode == "knee":
            ap_params = results.aperiodic_params
            if ap_params[1] <= 0:
                print(
                    "Negative value of knee parameter occurred. Suggestion: Fit without knee parameter."
                )
            knee_freq = np.abs(ap_params[1]) ** (1 / ap_params[2])
            print(f"Knee location: {knee_freq:.2f} Hz")

    if plot:
        plt_range = set_range(plt_range)
        fm.plot(plt_log=plt_log)
        plt.xlim(np.log10(plt_range) if plt_log else plt_range)
        # plt.ylim(-8, -5.5)
        if figsize:
            plt.gcf().set_size_inches(figsize)
        if title:
            plt.title(title)
        if is_notebook():
            pass
        else:
            plt.show()

    return results, fm


def generate_resd_from_fooof(fooof_model: FOOOF) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate residuals from a fitted FOOOF model.

    Parameters
    ----------
    fooof_model : FOOOF
        A fitted FOOOF model object.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing the residual power spectral density and the aperiodic fit.
    """
    results = fooof_model.get_results()
    full_fit, _, ap_fit = gen_model(
        fooof_model.freqs[1:],
        results.aperiodic_params,
        results.gaussian_params,
        return_components=True,
    )

    full_fit, ap_fit = 10**full_fit, 10**ap_fit  # Convert back from log
    res_psd = np.insert(
        (10 ** fooof_model.power_spectrum[1:]) - ap_fit, 0, 0.0
    )  # Convert back from log
    res_fit = np.insert(full_fit - ap_fit, 0, 0.0)
    ap_fit = np.insert(ap_fit, 0, 0.0)

    return res_psd, ap_fit


def calculate_SNR(fooof_model: FOOOF, freq_band: Tuple[float, float]) -> float:
    """
    Calculate the signal-to-noise ratio (SNR) from a fitted FOOOF model.

    Parameters
    ----------
    fooof_model : FOOOF
        A fitted FOOOF model object.
    freq_band : Tuple[float, float]
        Frequency band (min, max) for SNR calculation.

    Returns
    -------
    float
        The calculated SNR for the specified frequency band.
    """
    periodic, ap = generate_resd_from_fooof(fooof_model)
    freq = fooof_model.freqs  # Get frequencies from model
    indices = (freq >= freq_band[0]) & (freq <= freq_band[1])  # Get only the band we care about
    band_periodic = periodic[indices]  # Filter based on band
    band_ap = ap[indices]  # Filter
    band_freq = freq[indices]  # Another filter
    periodic_power = np.trapz(band_periodic, band_freq)  # Integrate periodic power
    ap_power = np.trapz(band_ap, band_freq)  # Integrate aperiodic power
    normalized_power = periodic_power / ap_power  # Compute the SNR
    return normalized_power


def calculate_wavelet_passband(center_freq: float, bandwidth: float, threshold: float = 0.3) -> Tuple[float, float, float]:
    """
    Calculate the passband of a complex Morlet wavelet filter.

    Parameters
    ----------
    center_freq : float
        Center frequency (Hz) of the wavelet filter.
    bandwidth : float
        Bandwidth parameter of the wavelet filter.
    threshold : float, optional
        Power threshold to define the passband edges (default: 0.3).

    Returns
    -------
    Tuple[float, float, float]
        (lower_bound, upper_bound, passband_width) of the frequency passband in Hz.
    """
    # Create a high-resolution frequency axis around the center frequency
    # Extend range to 3x the expected width to ensure we capture the full passband
    expected_width = center_freq * bandwidth / 2
    freq_min = max(0.1, center_freq - 3 * expected_width)
    freq_max = center_freq + 3 * expected_width
    freq_axis = np.linspace(freq_min, freq_max, 1000)

    # Calculate the theoretical frequency response of the Morlet wavelet
    # For a complex Morlet wavelet, the frequency response approximates a Gaussian
    # centered at the center frequency with width related to the bandwidth parameter
    sigma_f = bandwidth * center_freq / 8  # Approximate relationship for cmor wavelet
    response = np.exp(-((freq_axis - center_freq) ** 2) / (2 * sigma_f**2))

    # Find the passband edges (where response crosses the threshold)
    above_threshold = response >= threshold
    if not np.any(above_threshold):
        return (center_freq, center_freq, 0)  # No passband found

    # Find the first and last indices where response is above threshold
    indices = np.where(above_threshold)[0]
    lower_idx = indices[0]
    upper_idx = indices[-1]

    # Get the corresponding frequencies
    lower_bound = freq_axis[lower_idx]
    upper_bound = freq_axis[upper_idx]
    passband_width = upper_bound - lower_bound

    return (lower_bound, upper_bound, passband_width)


def wavelet_filter(
    x: np.ndarray,
    freq: float,
    fs: float,
    bandwidth: float = 1.0,
    axis: int = -1,
    show_passband: bool = False,
) -> np.ndarray:
    """
    Compute the Continuous Wavelet Transform (CWT) for a specified frequency using a complex Morlet wavelet.

    Parameters
    ----------
    x : np.ndarray
        Input signal.
    freq : float
        Target frequency for the wavelet filter.
    fs : float
        Sampling frequency of the signal.
    bandwidth : float, optional
        Bandwidth parameter of the wavelet filter (default is 1.0).
    axis : int, optional
        Axis along which to compute the CWT (default is -1).
    show_passband : bool, optional
        If True, print the passband of the wavelet filter (default is False).

    Returns
    -------
    np.ndarray
        Continuous Wavelet Transform of the input signal.
    """
    if show_passband:
        lower_bound, upper_bound, passband_width = calculate_wavelet_passband(
            freq, bandwidth, threshold=0.3
        )  # kinda made up threshold gives the rough idea
        print(f"Wavelet filter at {freq:.1f} Hz Bandwidth: {bandwidth:.1f} Hz:")
        print(
            f"  Passband: {lower_bound:.1f} - {upper_bound:.1f} Hz (width: {passband_width:.1f} Hz)"
        )
    wavelet = "cmor" + str(2 * bandwidth**2) + "-1.0"
    scale = pywt.scale2frequency(wavelet, 1) * fs / freq
    x_a = pywt.cwt(x, [scale], wavelet=wavelet, axis=axis)[0][0]
    return x_a


def butter_bandpass_filter(
    data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 5, axis: int = -1
) -> np.ndarray:
    """
    Apply a Butterworth bandpass filter to the input data.

    Parameters
    ----------
    data : np.ndarray
        Input data to filter.
    lowcut : float
        Lower cutoff frequency (Hz).
    highcut : float
        Upper cutoff frequency (Hz).
    fs : float
        Sampling frequency (Hz).
    order : int, optional
        Order of the Butterworth filter (default is 5).
    axis : int, optional
        Axis along which to apply the filter (default is -1).

    Returns
    -------
    np.ndarray
        Filtered data.
    """
    sos = signal.butter(order, [lowcut, highcut], fs=fs, btype="band", output="sos")
    x_a = signal.sosfiltfilt(sos, data, axis=axis)
    return x_a


def get_lfp_power(
    lfp_data: Union[np.ndarray, xr.DataArray],
    freq: float,
    fs: float,
    filter_method: str = "wavelet",
    lowcut: Optional[float] = None,
    highcut: Optional[float] = None,
    bandwidth: float = 1.0,
) -> Union[np.ndarray, xr.DataArray]:
    """
    Compute the power of the raw LFP signal in a specified frequency band,
    preserving xarray structure if input is xarray.

    Parameters
    ----------
    lfp_data : Union[np.ndarray, xr.DataArray]
        Raw local field potential (LFP) time series data.
    freq : float
        Center frequency (Hz) for wavelet filtering method.
    fs : float
        Sampling frequency (Hz) of the input data.
    filter_method : str, optional
        Filtering method to use, either 'wavelet' or 'butter' (default: 'wavelet').
    lowcut : Optional[float], optional
        Lower frequency bound (Hz) for butterworth bandpass filter, required if filter_method='butter'. Default is None.
    highcut : Optional[float], optional
        Upper frequency bound (Hz) for butterworth bandpass filter, required if filter_method='butter'. Default is None.
    bandwidth : float, optional
        Bandwidth parameter for wavelet filter when method='wavelet' (default: 1.0).

    Returns
    -------
    Union[np.ndarray, xr.DataArray]
        Power of the filtered signal (magnitude squared) with same structure as input.

    Notes
    -----
    - The 'wavelet' method uses a complex Morlet wavelet centered at the specified frequency
    - The 'butter' method uses a Butterworth bandpass filter with the specified cutoff frequencies
    - When using the 'butter' method, both lowcut and highcut must be provided
    - If input is an xarray DataArray, the output will preserve the same structure with coordinates
    """
    import xarray as xr

    # Check if input is xarray
    is_xarray = isinstance(lfp_data, xr.DataArray)

    if is_xarray:
        # Get the raw data from xarray
        raw_data = lfp_data.values
        # Check if 'fs' attribute exists in the xarray and override if necessary
        if "fs" in lfp_data.attrs and fs is None:
            fs = lfp_data.attrs["fs"]
    else:
        raw_data = lfp_data

    if filter_method == "wavelet":
        filtered_signal = wavelet_filter(raw_data, freq, fs, bandwidth)
    elif filter_method == "butter":
        if lowcut is None or highcut is None:
            raise ValueError(
                "Both lowcut and highcut must be specified when using 'butter' method."
            )
        filtered_signal = butter_bandpass_filter(raw_data, lowcut, highcut, fs)
    else:
        raise ValueError("Invalid method. Choose 'wavelet' or 'butter'.")

    # Calculate power (magnitude squared of filtered signal)
    power = np.abs(filtered_signal) ** 2

    # If the input was an xarray, return an xarray with the same coordinates
    if is_xarray:
        power_xarray = xr.DataArray(
            power,
            coords=lfp_data.coords,
            dims=lfp_data.dims,
            attrs={
                **lfp_data.attrs,
                "filter_method": filter_method,
                "frequency_of_interest": freq, # Changed from freq_of_interest
                "bandwidth": bandwidth,
                "lowcut": lowcut,
                "highcut": highcut,
                "power_type": "magnitude_squared",
            },
        )
        return power_xarray

    return power


def get_lfp_phase(
    lfp_data: Union[np.ndarray, xr.DataArray],
    freq: float,
    fs: float,
    filter_method: str = "wavelet",
    lowcut: Optional[float] = None,
    highcut: Optional[float] = None,
    bandwidth: float = 1.0,
) -> Union[np.ndarray, xr.DataArray]:
    """
    Calculate the phase of the filtered signal, preserving xarray structure if input is xarray.

    Parameters
    ----------
    lfp_data : Union[np.ndarray, xr.DataArray]
        Input LFP data.
    freq : float
        Frequency of interest (Hz).
    fs : float
        Sampling frequency (Hz).
    filter_method : str, optional
        Method for filtering the signal ('wavelet' or 'butter'), (default: 'wavelet').
    bandwidth : float, optional
        Bandwidth parameter for wavelet filter when method='wavelet' (default: 1.0).
    lowcut : Optional[float], optional
        Low cutoff frequency for Butterworth filter when method='butter'. Default is None.
    highcut : Optional[float], optional
        High cutoff frequency for Butterworth filter when method='butter'. Default is None.

    Returns
    -------
    Union[np.ndarray, xr.DataArray]
        Phase of the filtered signal with same structure as input.

    Notes
    -----
    - The 'wavelet' method uses a complex Morlet wavelet centered at the specified frequency
    - The 'butter' method uses a Butterworth bandpass filter with the specified cutoff frequencies
      followed by Hilbert transform to extract the phase
    - When using the 'butter' method, both lowcut and highcut must be provided
    - If input is an xarray DataArray, the output will preserve the same structure with coordinates
    """
    import xarray as xr

    # Check if input is xarray
    is_xarray = isinstance(lfp_data, xr.DataArray)

    if is_xarray:
        # Get the raw data from xarray
        raw_data = lfp_data.values
        # Check if 'fs' attribute exists in the xarray and override if necessary
        if "fs" in lfp_data.attrs and fs is None:
            fs = lfp_data.attrs["fs"]
    else:
        raw_data = lfp_data

    if filter_method == "wavelet":
        if freq is None:
            raise ValueError("freq must be provided for the wavelet method.")
        # Wavelet filter returns complex values directly
        filtered_signal = wavelet_filter(raw_data, freq, fs, bandwidth)
        # Phase is the angle of the complex signal
        phase = np.angle(filtered_signal)
    elif filter_method == "butter":
        if lowcut is None or highcut is None:
            raise ValueError(
                "Both lowcut and highcut must be specified when using 'butter' method."
            )
        # Butterworth filter returns real values
        filtered_signal = butter_bandpass_filter(raw_data, lowcut, highcut, fs)
        # Apply Hilbert transform to get analytic signal (complex)
        analytic_signal = signal.hilbert(filtered_signal)
        # Phase is the angle of the analytic signal
        phase = np.angle(analytic_signal)
    else:
        raise ValueError(f"Invalid method {filter_method}. Choose 'wavelet' or 'butter'.")

    # If the input was an xarray, return an xarray with the same coordinates
    if is_xarray:
        phase_xarray = xr.DataArray(
            phase,
            coords=lfp_data.coords,
            dims=lfp_data.dims,
            attrs={
                **lfp_data.attrs,
                "filter_method": filter_method,
                "freq_of_interest": freq, # Changed from freq_of_interest
                "bandwidth": bandwidth,
                "lowcut": lowcut,
                "highcut": highcut,
            },
        )
        return phase_xarray

    return phase


# windowing functions
def windowed_xarray(
    da: xr.DataArray,
    windows: np.ndarray,
    dim: str = "time",
    new_coord_name: str = "cycle",
    new_coord: Optional[pd.Index] = None,
) -> xr.DataArray:
    """Divide xarray into windows of equal size along an axis.

    Parameters
    ----------
    da : xr.DataArray
        Input DataArray.
    windows : np.ndarray
        2D array of windows, where each row is [start, stop].
    dim : str, optional
        Dimension along which to divide (default is "time").
    new_coord_name : str, optional
        Name of new dimension along which to concatenate windows (default is "cycle").
    new_coord : Optional[pd.Index], optional
        Pandas Index object of new coordinates. Defaults to integer index.

    Returns
    -------
    xr.DataArray
        Windowed DataArray.
    """
    win_da_list = [da.sel({dim: slice(*w)}) for w in windows]
    if not win_da_list: # Handle empty windows input
        return xr.DataArray() # Or raise error, or return empty with original dims
        
    n_win = min(x.coords[dim].size for x in win_da_list)
    idx = {dim: slice(n_win)}
    # Ensure coords are taken from the original da to avoid issues with empty win_da_list
    # However, this specific coords line might be problematic if n_win is 0.
    # It's better to construct coords based on the expected size `n_win`.
    if n_win > 0:
        coords_template = da.coords[dim].isel({dim: slice(n_win)})
        processed_win_da = [x.isel(idx).assign_coords({dim: coords_template}) for x in win_da_list]
    else: # if all windows resulted in zero length after selection or n_win is 0
        # Return an empty DataArray, possibly with original non-dim coords
        # This part might need adjustment based on desired behavior for empty windows
        return xr.DataArray(np.empty(tuple(0 if d == dim else da.sizes[d] for d in da.dims)),
                            coords={k: v for k,v in da.coords.items() if k != dim}, # Keep non-dim coords
                            dims=da.dims, name=da.name)


    if new_coord is None:
        new_coord = pd.Index(range(len(processed_win_da)), name=new_coord_name)
    concatenated_win_da = xr.concat(processed_win_da, dim=new_coord)
    return concatenated_win_da


def group_windows(
    win_da: xr.DataArray, win_grp_idx: Dict = {}, win_dim: str = "cycle"
) -> Tuple[Dict, Dict]:
    """Group windows into a dictionary of DataArrays.

    Parameters
    ----------
    win_da : xr.DataArray
        Input windowed DataArray.
    win_grp_idx : Dict, optional
        Dictionary of {window group id: window indices} (default is {}).
    win_dim : str, optional
        Dimension for different windows (default is "cycle").

    Returns
    -------
    Tuple[Dict, Dict]
        Dictionaries of {window group id: DataArray of grouped windows}
        win_on / win_off for windows selected / not selected by `win_grp_idx`.
    """
    win_on: Dict = {}
    win_off: Dict = {}
    for g, w_indices in win_grp_idx.items(): # renamed w to w_indices for clarity
        win_on[g] = win_da.sel({win_dim: w_indices})
        win_off[g] = win_da.drop_sel({win_dim: w_indices})
    return win_on, win_off


def average_group_windows(
    win_da: Dict, win_dim: str = "cycle", grp_dim: str = "unique_cycle"
) -> xr.Dataset:
    """Average over windows in each group and stack groups in a DataArray.

    Parameters
    ----------
    win_da : Dict
        Input dictionary of {window group id: DataArray of grouped windows}.
    win_dim : str, optional
        Dimension for different windows (default is "cycle").
    grp_dim : str, optional
        Dimension along which to stack average of window groups (default is "unique_cycle").

    Returns
    -------
    xr.Dataset
        Dataset containing mean and std of grouped windows.
    """
    win_avg = {
        g: xr.concat(
            [x.mean(dim=win_dim), x.std(dim=win_dim)], pd.Index(("mean_", "std_"), name="stats")
        )
        for g, x in win_da.items()
    }
    win_avg = xr.concat(win_avg.values(), dim=pd.Index(win_avg.keys(), name=grp_dim))
    win_avg = win_avg.to_dataset(dim="stats")
    return win_avg


# used for avg spectrogram across different trials
def get_windowed_data(
    x: xr.DataArray,
    windows: np.ndarray,
    win_grp_idx: Dict,
    dim: str = "time",
    win_dim: str = "cycle",
    win_coord: Optional[pd.Index] = None,
    grp_dim: Optional[str] = "unique_cycle",
) -> Tuple[xr.DataArray, Tuple[Dict, Dict], Optional[List[xr.Dataset]]]:
    """Apply functions of windowing to data.

    Parameters
    ----------
    x : xr.DataArray
        Input DataArray.
    windows : np.ndarray
        2D array of windows for `windowed_xarray`.
    win_grp_idx : Dict
        Dictionary of {window group id: window indices} for `group_windows`.
    dim : str, optional
        Dimension along which to divide (default is "time").
    win_dim : str, optional
        Dimension for different windows (default is "cycle").
    win_coord : Optional[pd.Index], optional
        Pandas Index object of `win_dim` coordinates.
    grp_dim : Optional[str], optional
        Dimension along which to stack average of window groups.
        If None, averaging is skipped (default is "unique_cycle").

    Returns
    -------
    Tuple[xr.DataArray, Tuple[Dict, Dict], Optional[List[xr.Dataset]]]
        Data returned by three functions:
        - `windowed_xarray` output
        - `group_windows` output (a tuple of two dicts)
        - List of `average_group_windows` outputs (or None if grp_dim is None)
    """
    x_win = windowed_xarray(x, windows, dim=dim, new_coord_name=win_dim, new_coord=win_coord)
    x_win_on_off = group_windows(x_win, win_grp_idx, win_dim=win_dim) # Renamed for clarity
    x_win_avg: Optional[List[xr.Dataset]] = None
    if grp_dim:
        # Applying average_group_windows to each of the two dictionaries returned by group_windows
        x_win_avg = [average_group_windows(d, win_dim=win_dim, grp_dim=grp_dim) for d in x_win_on_off if isinstance(d, dict)]
    return x_win, x_win_on_off, x_win_avg


# cone of influence in frequency for cmorxx-1.0 wavelet. need to add logic to calculate in function
f0 = 2 * np.pi
CMOR_COI = 2**-0.5
CMOR_FLAMBDA = 4 * np.pi / (f0 + (2 + f0**2) ** 0.5)
COI_FREQ = 1 / (CMOR_COI * CMOR_FLAMBDA)


def cwt_spectrogram(
    x: np.ndarray,
    fs: float,
    n_notes: int = 6,
    n_octaves: float = np.inf,
    freq_range: Tuple[float, float] = (0, np.inf),
    bandwidth: float = 1.0,
    axis: int = -1,
    detrend: bool = False,
    normalize: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate spectrogram using continuous wavelet transform.

    Parameters
    ----------
    x : np.ndarray
        Input signal.
    fs : float
        Sampling frequency (Hz).
    n_notes : int, optional
        Number of notes per octave for scale resolution (default is 6).
    n_octaves : float, optional
        Number of octaves to analyze (default is np.inf, adjusted to data length).
    freq_range : Tuple[float, float], optional
        Frequency range (min_freq, max_freq) to output (default is (0, np.inf)).
    bandwidth : float, optional
        Bandwidth parameter of the complex Morlet wavelet (default is 1.0).
    axis : int, optional
        Axis along which to compute the CWT (default is -1).
    detrend : bool, optional
        If True, detrend the signal before CWT (default is False).
    normalize : bool, optional
        If True, normalize the signal before CWT (default is False).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        - power : np.ndarray - Spectrogram power.
        - times : np.ndarray - Time vector for the spectrogram.
        - frequencies : np.ndarray - Frequency vector for the spectrogram.
        - coif : np.ndarray - Cone of influence frequencies.
    """
    x_processed = np.asarray(x) # Use a different variable name to avoid modifying input x if it's passed by reference elsewhere
    N = x_processed.shape[axis]
    times = np.arange(N) / fs
    # detrend and normalize
    if detrend:
        x_processed = signal.detrend(x_processed, axis=axis, type="linear")
    if normalize:
        x_processed = x_processed / x_processed.std() # make sure std is not zero
    # Define some parameters of our wavelet analysis.
    # range of scales (in time) that makes sense
    # min = 2 (Nyquist frequency)
    # max = np.floor(N/2)
    n_octaves_eff = min(n_octaves, np.log2(2 * np.floor(N / 2))) # Effective n_octaves
    scales = 2 ** np.arange(1, n_octaves_eff, 1 / n_notes)
    # cwt and the frequencies used.
    # Use the complex morelet with bw=2*bandwidth^2 and center frequency of 1.0
    # bandwidth is sigma of the gaussian envelope
    wavelet_str = "cmor" + str(2 * bandwidth**2) + "-1.0" # Renamed variable
    frequencies = pywt.scale2frequency(wavelet_str, scales) * fs
    # Filter scales based on frequency range
    valid_freq_indices = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
    scales = scales[valid_freq_indices]
    frequencies = frequencies[valid_freq_indices]

    if len(scales) == 0: # Handle case where no scales match the frequency range
        # Return empty arrays or raise an error, depending on desired behavior
        # For now, returning empty arrays of appropriate dimensions
        empty_power_shape = list(x_processed.shape)
        empty_power_shape.insert(axis if axis >=0 else len(x_processed.shape)+axis , 0) # insert freq dim
        
        # Adjust shape for output of cwt which swaps axis and new freq dim
        # if axis is -1, freq dim is at -2. if axis is 0, freq dim is at 0.
        # pywt.cwt output shape: (n_scales, ...) if axis=0, or (..., n_scales, ...) if axis=-1
        # The current code moves the scale/freq axis to -2 later.
        # For power, it should be (..., 0, N)
        power_shape_final = list(x_processed.shape)
        freq_axis_pos = axis if axis >=0 else len(x_processed.shape)+axis
        if freq_axis_pos < 0: freq_axis_pos = len(x_processed.shape) + freq_axis_pos # Ensure positive index
        
        # The power output shape after np.moveaxis(coef, 0, -2) would be (..., 0, N)
        # N = x_processed.shape[axis]
        # other_dims = [d for i, d in enumerate(x_processed.shape) if i != N_axis_abs]
        # final_power_shape = tuple(other_dims) + (0, N)

        # Simplified: let's figure out the shape based on how coef is handled
        # coef shape from pywt.cwt: (len(scales), N) when axis=-1 and x is 1D
        # or (len(scales), M, N) if x is (M,N) and axis=-1
        # power shape will be same as coef.
        # The code has `coef, frequencies = pywt.cwt(...)`
        # then `power = np.real(coef * np.conj(coef))`
        # If scales is empty, coef will likely be empty or error.
        # pywt.cwt might error if scales is empty. Let's assume it returns empty array.
        # For safety, if frequencies is empty, return empty arrays.
        coif = np.array([]) # Default empty coif
        if len(frequencies) == 0:
            power_dims = list(x_processed.shape)
            # remove original time axis, add new freq and time axes
            # This is tricky because pywt.cwt changes dimensionality.
            # For now, return empty arrays with a warning or log.
            # This part needs careful consideration of expected output shapes for empty inputs.
            # Returning simple empty arrays for now.
            print("Warning: No scales matched the frequency range. Returning empty spectrogram.")
            return np.array([]), times, np.array([]), coif


    coef, frequencies = pywt.cwt(
        x_processed, scales[::-1], wavelet=wavelet_str, sampling_period=1 / fs, axis=axis
    )
    power = np.real(coef * np.conj(coef))  # equivalent to power = np.abs(coef)**2
    # cone of influence in terms of wavelength
    coi = N / 2 - np.abs(np.arange(N) - (N - 1) / 2)
    # cone of influence in terms of frequency
    coif = COI_FREQ * fs / coi
    return power, times, frequencies, coif


def cwt_spectrogram_xarray(
    x: np.ndarray,
    fs: float,
    time: Optional[np.ndarray] = None,
    axis: int = -1,
    downsample_fs: Optional[float] = None,
    channel_coords: Optional[Dict[str, Union[list, np.ndarray]]] = None,
    **cwt_kwargs: Dict,
) -> xr.Dataset:
    """Calculate spectrogram using continuous wavelet transform and return an xarray.Dataset.

    Parameters
    ----------
    x : np.ndarray
        Input array.
    fs : float
        Sampling frequency (Hz).
    time : Optional[np.ndarray], optional
        Time vector corresponding to the time axis in x. If None, it's computed. Default is None.
    axis : int, optional
        Dimension index of time axis in x (default is -1).
    downsample_fs : Optional[float], optional
        Frequency to downsample to. If None, no downsampling (default is None).
    channel_coords : Optional[Dict[str, Union[list, np.ndarray]]], optional
        Dictionary of {coordinate name: index} for channels (default is None).
    **cwt_kwargs : Dict
        Keyword arguments for cwt_spectrogram().

    Returns
    -------
    xr.Dataset
        Dataset containing the PSD and cone of influence.
    """
    x = np.asarray(x)
    T = x.shape[axis]  # number of time points
    t = np.arange(T) / fs if time is None else np.asarray(time)
    if downsample_fs is None or downsample_fs >= fs:
        downsample_fs = fs
        downsampled = x
    else:
        num = int(T * downsample_fs / fs)
        downsample_fs = num / T * fs
        downsampled, t = signal.resample(x, num=num, t=t, axis=axis)
    downsampled = np.moveaxis(downsampled, axis, -1)
    sxx, _, f, coif = cwt_spectrogram(downsampled, downsample_fs, **cwt_kwargs)
    sxx = np.moveaxis(sxx, 0, -2)  # shape (... , freq, time)
    if channel_coords is None:
        channel_coords = {f"dim_{i:d}": range(d) for i, d in enumerate(sxx.shape[:-2])}
    sxx = xr.DataArray(sxx, coords={**channel_coords, "frequency": f, "time": t}).to_dataset(
        name="PSD"
    )
    sxx.update(dict(cone_of_influence_frequency=xr.DataArray(coif, coords={"time": t})))
    return sxx
