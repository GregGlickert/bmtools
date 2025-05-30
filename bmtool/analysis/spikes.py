"""
Module for processing BMTK spikes output.
"""

import os
from typing import List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import mannwhitneyu

from bmtool.util.util import load_nodes_from_config


def load_spikes_to_df(
    spike_file: str,
    network_name: str,
    sort: bool = True,
    config_path: Optional[str] = None, # Renamed from config and made Optional
    groupby: Union[str, List[str]] = "pop_name",
) -> pd.DataFrame:
    """
    Load spike data from an HDF5 file into a pandas DataFrame.

    Parameters
    ----------
    spike_file : str
        Path to the HDF5 file containing spike data.
    network_name : str
        The name of the network within the HDF5 file from which to load spike data.
    sort : bool, optional
        Whether to sort the DataFrame by 'timestamps' (default is True).
    config_path : Optional[str], optional
        Path to configuration file to label the cell type of each spike (default is None).
    groupby : Union[str, List[str]], optional
        The column(s) to group by (default is 'pop_name').

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing 'node_ids' and 'timestamps' columns from the spike data,
        with additional columns if a config_path file is provided.

    Examples
    --------
    >>> df = load_spikes_to_df("spikes.h5", "cortex")
    >>> df = load_spikes_to_df("spikes.h5", "cortex", config_path="config.json", groupby=["pop_name", "model_type"])
    """
    with h5py.File(spike_file) as f:
        spikes_df = pd.DataFrame(
            {
                "node_ids": f["spikes"][network_name]["node_ids"],
                "timestamps": f["spikes"][network_name]["timestamps"],
            }
        )

        if sort:
            spikes_df.sort_values(by="timestamps", inplace=True, ignore_index=True)

        if config_path:
            nodes = load_nodes_from_config(config_path)
            nodes = nodes[network_name] # Assuming network_name is a key in the dict returned by load_nodes_from_config

            # Convert single string to a list for uniform handling
            if isinstance(groupby, str):
                groupby = [groupby]

            # Ensure all requested columns exist
            missing_cols = [col for col in groupby if col not in nodes.columns]
            if missing_cols:
                raise KeyError(f"Columns {missing_cols} not found in nodes DataFrame.")

            spikes_df = spikes_df.merge(
                nodes[groupby], left_on="node_ids", right_index=True, how="left"
            )

    return spikes_df


def compute_firing_rate_stats(
    df: pd.DataFrame,
    groupby: Union[str, List[str]] = "pop_name",
    t_start: Optional[float] = None,      # Renamed from start_time
    t_stop: Optional[float] = None,       # Renamed from stop_time
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes the firing rates of individual nodes and the mean and standard deviation of firing rates per group.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing spike timestamps and node IDs.
        Must include 'timestamps', 'node_ids', and columns specified in `groupby`.
    groupby : Union[str, List[str]], optional
        Column(s) to group by (e.g., 'pop_name' or ['pop_name', 'layer']). Default is 'pop_name'.
    t_start : Optional[float], optional
        Start time (ms) for the analysis window. Defaults to the minimum timestamp in the data.
    t_stop : Optional[float], optional
        Stop time (ms) for the analysis window. Defaults to the maximum timestamp in the data.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - pop_stats : pd.DataFrame
            Contains the mean and standard deviation of firing rates per group.
        - individual_stats : pd.DataFrame
            Contains the firing rate (Hz) of each individual node.
    """

    # Ensure groupby is a list
    if isinstance(groupby, str):
        groupby = [groupby]

    # Ensure all columns exist in the dataframe
    for col in groupby:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe.")

    # Filter dataframe based on start/stop time
    if t_start is not None:
        df = df[df["timestamps"] >= t_start]
    if t_stop is not None:
        df = df[df["timestamps"] <= t_stop]

    # Compute total duration for firing rate calculation
    min_time_val = df["timestamps"].min() if t_start is None else t_start
    max_time_val = df["timestamps"].max() if t_stop is None else t_stop

    duration = max_time_val - min_time_val  # Duration in ms

    if duration <= 0: # Check after correct assignment
        # If t_start and t_stop are the same, or filtered df is empty, duration might be zero or negative.
        # Return empty DataFrames or raise error, as firing rate is undefined.
        # For now, let's align with the original code's implicit behavior of erroring out later
        # or returning NaNs/zeros if spike_counts is empty.
        # A more robust solution might return empty DFs with correct columns.
        # However, the original code would lead to division by zero if duration is 0.
        # Let's ensure a ValueError is raised for non-positive duration.
        raise ValueError(
            "Time window duration must be positive. "
            f"Calculated duration: {duration} ms from min_time_val: {min_time_val} and max_time_val: {max_time_val}."
        )

    # Compute firing rate for each node

    # Compute spike counts per node
    spike_counts = df["node_ids"].value_counts().reset_index()
    spike_counts.columns = ["node_ids", "spike_count"]  # Rename columns

    # Merge with original dataframe to get corresponding labels (e.g., 'pop_name')
    spike_counts = spike_counts.merge(
        df[["node_ids"] + groupby].drop_duplicates(), on="node_ids", how="left"
    )

    # Compute firing rate
    spike_counts["firing_rate"] = spike_counts["spike_count"] / (duration / 1000.0)  # scale to Hz (duration to seconds)
    individual_stats = spike_counts # Corrected typo

    # Compute mean and standard deviation per group
    pop_stats = individual_stats.groupby(groupby)["firing_rate"].agg(["mean", "std"]).reset_index()

    # Rename columns
    pop_stats.rename(columns={"mean": "firing_rate_mean", "std": "firing_rate_std"}, inplace=True)

    return pop_stats, individual_stats


def _pop_spike_rate(
    spike_times: Union[np.ndarray, List[float]], # Changed list to List[float]
    time: Optional[Tuple[float, float, float]] = None,
    time_points: Optional[Union[np.ndarray, List[float]]] = None, # Changed list to List[float]
    frequency: bool = False,
) -> np.ndarray:
    """
    Calculate the spike count or frequency histogram over specified time intervals.

    Parameters
    ----------
    spike_times : Union[np.ndarray, List[float]]
        Array or list of spike times in milliseconds.
    time : Optional[Tuple[float, float, float]], optional
        Tuple specifying (start_time, stop_time, step_ms) in milliseconds.
        Used to create evenly spaced time points if `time_points` is not provided. Default is None.
    time_points : Optional[Union[np.ndarray, List[float]]], optional
        Array or list of specific time points (bin edges) for binning.
        If provided, `time` is ignored. Default is None.
    frequency : bool, optional
        If True, returns spike frequency in Hz; otherwise, returns spike count. Default is False.

    Returns
    -------
    np.ndarray
        Array of spike counts or frequencies, corresponding to the bins defined by `time_points` or `time`.

    Raises
    ------
    ValueError
        If both `time` and `time_points` are None, or if `dt` (time step) is zero or negative.
    """
    if time_points is None:
        if time is None:
            raise ValueError("Either `time` or `time_points` must be provided.")
        if time is None:
            raise ValueError("Either `time` or `time_points` must be provided.")
        time_points_np = np.arange(*time) # Ensure it's a numpy array for consistent processing
        dt = time[2] # time step in ms
    else:
        time_points_np = np.asarray(time_points).ravel()
        if time_points_np.size < 2:
            raise ValueError("`time_points` must contain at least two points to define a bin.")
        # dt is calculated as the average difference between consecutive time points if not fixed by `time`
        # However, for histogram, the bins are what matters.
        # The dt for frequency calculation should represent the width of the bins.
        # If time_points are bin edges, then dt is the difference. If they are bin centers, it's more complex.
        # Assuming time_points are bin edges (as np.histogram expects bin edges).
        # For frequency calculation, using the first bin width as representative if bins are uneven.
        dt = time_points_np[1] - time_points_np[0] # More robust for frequency calculation if bins are uniform

    if dt <= 0:
        raise ValueError("Time step `dt` must be positive.")

    # Ensure bins cover the entire range specified by time_points
    # If time_points are considered bin edges, np.histogram will use them directly.
    # The original code `bins = np.append(time_points, time_points[-1] + dt)` assumes time_points are starts of bins.
    # Let's clarify: np.histogram `bins` argument can be an int (number of bins) or array (bin edges).
    # If `time_points` are the bin edges:
    bins = time_points_np
    
    spike_counts, _ = np.histogram(np.asarray(spike_times), bins=bins)

    if frequency:
        # dt for frequency should be the width of the bins.
        # If bins are not uniform, this needs careful handling.
        # Assuming uniform bin width based on the first two points of `time_points` or `time[2]`.
        bin_widths = np.diff(bins) # Width of each bin
        if not np.allclose(bin_widths, bin_widths[0]): # Check if all bin widths are the same
             print(f"Warning: Bin widths are not uniform. Using mean bin width ({np.mean(bin_widths)} ms) for frequency calculation.")
             # Using mean bin width for frequency calculation if bins are not uniform.
             # This might not be ideal for all cases but is a reasonable compromise.
             effective_dt_for_freq_calc = np.mean(bin_widths)
        else:
            effective_dt_for_freq_calc = bin_widths[0]

        if effective_dt_for_freq_calc <= 0:
             raise ValueError("Bin width `dt` for frequency calculation must be positive.")
        spike_rate_freq = spike_counts / (effective_dt_for_freq_calc / 1000.0) # Convert ms to s for Hz
        return spike_rate_freq
    
    return spike_counts


def get_population_spike_rate(
    spike_data: pd.DataFrame,
    fs: float = 400.0,
    t_start: float = 0,
    t_stop: Optional[float] = None,
    config_path: Optional[str] = None, # Renamed from config
    network_name: Optional[str] = None,
    save: bool = False,
    save_path: Optional[str] = None,
    normalize: bool = False,
    smooth_window: int = 50,  # Window size for smoothing (in time bins)
    smooth_method: str = "gaussian",  # Smoothing method: 'gaussian', 'boxcar', or 'exponential'
) -> xr.DataArray:
    """
    Calculate the population spike rate for each population in the given spike data.

    Parameters
    ----------
    spike_data : pd.DataFrame
        A DataFrame containing spike data with columns 'pop_name', 'timestamps', and 'node_ids'
    fs : float, optional
        Sampling frequency in Hz, which determines the time bin size for calculating the spike rate (default: 400.0)
    t_start : float, optional
        Start time (in milliseconds) for spike rate calculation (default: 0)
    t_stop : Optional[float], optional
        Stop time (in milliseconds) for spike rate calculation. If None, defaults to the maximum timestamp in the data
    config_path : Optional[str], optional
        Path to a configuration file containing node information, used to determine the correct number of nodes per population.
        If None, node count is estimated from unique node spikes (default: None).
    network_name : Optional[str], optional
        Name of the network used in the configuration file, allowing selection of nodes for that network.
        Required if `config_path` is provided (default: None).
    save : bool, optional
        Whether to save the calculated population spike rate to a file (default: False)
    save_path : Optional[str], optional
        Directory path where the file should be saved if `save` is True (default: None)
    normalize : bool, optional
        Whether to normalize the spike rates for each population to a range of [0, 1] (default: False)
    smooth_window : int, optional
        Window size for smoothing in number of time bins (default: 50)
    smooth_method : str, optional
        Smoothing method to use: 'gaussian', 'boxcar', or 'exponential' (default: 'gaussian')

    Returns
    -------
    xr.DataArray
        An xarray DataArray containing the spike rates with dimensions of time, population, and type.
        The 'type' dimension includes 'raw' and 'smoothed' values.
        The DataArray includes sampling frequency (fs) as an attribute.
        If normalize is True, each population's spike rate is scaled to [0, 1].

    Raises
    ------
    ValueError
        If `save` is True but `save_path` is not provided.
        If an invalid smooth_method is specified.

    Notes
    -----
    - If `config_path` is None, the function assumes all cells in each population have fired at least once;
      otherwise, the node count may be inaccurate.
    - If normalization is enabled, each population's spike rate is scaled using Min-Max normalization.
    - Smoothing is applied using scipy.ndimage's filters based on the specified method.
    """
    import numpy as np
    from scipy import ndimage

    # Validate smoothing method
    if smooth_method not in ["gaussian", "boxcar", "exponential"]:
        raise ValueError(
            f"Invalid smooth_method: {smooth_method}. Choose from 'gaussian', 'boxcar', or 'exponential'."
        )

    pop_spikes: Dict[str, pd.DataFrame] = {}
    node_number: Dict[str, int] = {}

    if config_path is None:
        print(
            "Note: Node number is obtained by counting unique node spikes in the network.\nIf the network did not run for a sufficient duration, or not all cells fired,\nthen this count will not include all nodes so the firing rate will not be of the whole population!"
        )
        print(
            "You can provide a config_path to calculate the correct amount of nodes! for a true population rate."
        )

    if config_path:
        if not network_name:
            # This print statement is for user info, could be a warning log too.
            print(
                "Warning: `config_path` provided but `network_name` is None. "
                "Attempting to use the first network found in the config, which may not be desired."
            )

    # Get t_stop if not provided
    if t_stop is None:
        t_stop = spike_data["timestamps"].max()

    # Get population names and prepare data
    populations = spike_data["pop_name"].unique()
    for pop_name in populations:
        pop_specific_spikes_df = spike_data[spike_data["pop_name"] == pop_name]

        if config_path:
            # nodes_config is expected to be a dict of DataFrames (one per network)
            nodes_config = load_nodes_from_config(config_path) # type: ignore
            if network_name and network_name in nodes_config:
                current_network_nodes: pd.DataFrame = nodes_config[network_name] # type: ignore
            elif not network_name and nodes_config:
                # Fallback to the first network if network_name is not specified
                current_network_nodes = list(nodes_config.values())[0] # type: ignore
            else:
                # This case means config_path was given, but network_name might be wrong or config empty
                raise ValueError(f"Network '{network_name}' not found in config or config is empty.")

            pop_nodes_in_config = current_network_nodes[current_network_nodes["pop_name"] == pop_name]
            node_number[pop_name] = pop_nodes_in_config.index.nunique()
        else:
            node_number[pop_name] = pop_specific_spikes_df["node_ids"].nunique()

        # Filter spikes for the current population and time window
        filtered_spikes_df = pop_specific_spikes_df[
            (pop_specific_spikes_df["timestamps"] >= t_start) & (pop_specific_spikes_df["timestamps"] <= t_stop)
        ]
        pop_spikes[pop_name] = filtered_spikes_df

    # Define time bins for histogramming based on fs, t_start, t_stop
    # np.arange might not include t_stop if (t_stop - t_start) is not a multiple of step
    # Using np.linspace to ensure the number of points corresponds to fs over the duration
    # The number of intervals will be fs * duration_seconds - 1, so num_points is fs * duration_seconds
    duration_ms = t_stop - t_start
    num_time_points = int(np.ceil(duration_ms / (1000.0 / fs))) + 1 # +1 to include both ends if using linspace for edges
    time_bin_edges = np.linspace(t_start, t_stop, num_time_points)
    # time_coords are usually bin centers or starts. For _pop_spike_rate, it expects bin edges.
    # The output of _pop_spike_rate will have len(time_bin_edges) - 1 elements.
    # So, the time coordinates for the output DataArray should be these bin centers or starts.
    time_coords_for_output = time_bin_edges[:-1] # Or (time_bin_edges[:-1] + time_bin_edges[1:]) / 2 for centers

    # Calculate spike rates for each population
    spike_rates_list: List[np.ndarray] = []
    for p_name in populations:
        # _pop_spike_rate expects spike times and bin edges
        # It returns counts, so frequency=False (default)
        # Spike counts per bin
        spike_counts_per_bin = _pop_spike_rate(
            pop_spikes[p_name]["timestamps"].values, time_points=time_bin_edges, frequency=False
        )
        
        # Rate = (spike_counts_per_bin / num_nodes_in_pop) / (bin_width_seconds)
        # bin_width_ms = 1000.0 / fs
        # rate = (spike_counts_per_bin / node_number[p_name]) / (bin_width_ms / 1000.0)
        # Simplified: rate = spike_counts_per_bin * fs / node_number[p_name]
        # This is average spikes per second per neuron in that population
        if node_number[p_name] == 0: # Avoid division by zero
            rate = np.zeros_like(spike_counts_per_bin, dtype=float)
        else:
            rate = spike_counts_per_bin * fs / node_number[p_name]
        spike_rates_list.append(rate)

    spike_rates_array = np.array(spike_rates_list).T  # Transpose to have time as first dimension

    # Calculate smoothed version for each population
    smoothed_rates = []

    for i in range(spike_rates_array.shape[1]):
        pop_rate = spike_rates_array[:, i]

        if smooth_method == "gaussian":
            # Gaussian smoothing (sigma is approximately window/6 for a Gaussian filter)
            sigma = smooth_window / 6
            smoothed_pop_rate = ndimage.gaussian_filter1d(pop_rate, sigma=sigma)
        elif smooth_method == "boxcar":
            # Boxcar/uniform smoothing
            kernel = np.ones(smooth_window) / smooth_window
            smoothed_pop_rate = ndimage.convolve1d(pop_rate, kernel, mode="nearest")
        elif smooth_method == "exponential":
            # Exponential smoothing
            alpha = 2 / (smooth_window + 1)  # Equivalent to window size in exponential smoothing
            smoothed_pop_rate = np.zeros_like(pop_rate)
            smoothed_pop_rate[0] = pop_rate[0]
            for t in range(1, len(pop_rate)):
                smoothed_pop_rate[t] = alpha * pop_rate[t] + (1 - alpha) * smoothed_pop_rate[t - 1]

        smoothed_rates.append(smoothed_pop_rate)

    smoothed_rates_array = np.array(smoothed_rates).T  # Transpose to have time as first dimension

    # Stack raw and smoothed data
    combined_data = np.stack([spike_rates_array, smoothed_rates_array], axis=2)

    # Create DataArray with the additional 'type' dimension
    spike_rate_xarray = xr.DataArray( # Renamed for clarity
        combined_data,
        coords={"time": time_coords_for_output, "population": populations, "type": ["raw", "smoothed"]},
        dims=["time", "population", "type"],
        attrs={
            "fs": fs, # Original sampling rate of spike data, or effective sampling rate of the bins
            "description": "Population spike rate (Hz per neuron).",
            "normalized": False,
            "smooth_method": smooth_method,
            "smooth_window_bins": smooth_window, # Clarify unit of smooth_window
        },
    )

    # Normalize if requested
    if normalize:
        # Apply normalization for each population and each type (raw/smoothed)
        for pop_label in populations: # Iterate by label for clarity
            for type_label in ["raw", "smoothed"]: # Iterate by label
                pop_data = spike_rate_xarray.sel(population=pop_label, type=type_label)
                min_val = pop_data.min(dim="time")
                max_val = pop_data.max(dim="time")

                # Handle case where min == max (constant signal or single point)
                if max_val > min_val: # Ensure max_val is strictly greater than min_val
                    spike_rate_xarray.loc[dict(population=pop_label, type=type_label)] = (pop_data - min_val) / (max_val - min_val)
                elif max_val == min_val and min_val != 0 : # If constant non-zero, normalize to 0.5 or 1? Or leave as is?
                    # Typically, if min=max, data is normalized to 0 or 0.5. Let's choose 0.
                    spike_rate_xarray.loc[dict(population=pop_label, type=type_label)] = xr.zeros_like(pop_data)


        spike_rate_xarray.attrs["normalized"] = True

    # Save if requested
    if save:
        if save_path is None:
            raise ValueError("save_path must be provided if save is True.")

        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, "population_spike_rate.nc") # Changed extension to .nc for netCDF
        spike_rate_xarray.to_netcdf(save_file)

    return spike_rate_xarray


def average_spike_rate_over_windows(
    spike_rate: xr.DataArray, windows: List[Tuple[float, float]]
) -> xr.DataArray:
    """
    Calculate the average spike rate over multiple time windows.

    Parameters
    ----------
    spike_rate : xr.DataArray
        The spike rate data array, typically with dimensions (time, population, [type]),
        where 'type' can be 'raw' or 'smoothed'.
    windows : List[Tuple[float, float]]
        List of (start_time_ms, end_time_ms) tuples defining the windows to average over.

    Returns
    -------
    xr.DataArray
        Averaged spike rate. The time dimension is normalized to start at 0 for each window's duration.
        Other dimensions (like population, type) are preserved.
    """
    # Check if the DataArray has a 'type' dimension (compatible with new format)
    has_type_dim = "type" in spike_rate.dims

    # Initialize list to store data from each window
    window_data = []

    # Get data for each window
    for start, end in windows:
        # Select data points within the window
        window = spike_rate.sel(time=slice(start, end))

        # Normalize time to start at 0 for this window
        window = window.assign_coords(time=window.time - start)
        window_data.append(window)

    # Align and average windows
    # First window determines the time coordinates
    aligned_data = xr.concat(window_data, dim="window")
    averaged_data = aligned_data.mean(dim="window")

    # Create new DataArray with the averaged data
    if has_type_dim:
        # Create result with time, population, and type dimensions
        result = xr.DataArray(
            averaged_data.values,
            coords={
                "time": averaged_data.time.values,
                "population": averaged_data.population,
                "type": averaged_data.type,
            },
            dims=["time", "population", "type"],
        )
    else:
        # Handle older format without 'type' dimension (for backward compatibility)
        result = xr.DataArray(
            averaged_data.values,
            coords={"time": averaged_data.time.values, "population": averaged_data.population},
            dims=["time", "population"],
        )

    # Preserve attributes
    result.attrs = spike_rate.attrs

    return result


def compare_firing_over_times(
    spike_df: pd.DataFrame,
    group_by: str,
    time_window_1: Tuple[float, float], # Changed to Tuple
    time_window_2: Tuple[float, float], # Changed to Tuple
) -> None:
    """
    Compares the firing rates of a population during two different time windows and performs
    a statistical test to determine if there is a significant difference.

    Parameters
    ----------
    spike_df : pd.DataFrame
        DataFrame containing spike data with columns for 'timestamps', 'node_ids', and the `group_by` column.
    group_by : str
        Column name to group spikes by (e.g., 'pop_name').
    time_window_1 : Tuple[float, float]
        First time window as (start_ms, stop_ms).
    time_window_2 : Tuple[float, float]
        Second time window as (start_ms, stop_ms).

    Returns
    -------
    None
        Results are printed to the console.

    Notes
    -----
    Uses Mann-Whitney U test (non-parametric) to compare firing rates between the two windows.
    """
    # Filter spikes for the population of interest
    for pop_name in spike_df[group_by].unique():
        print(f"Population: {pop_name}")
        pop_spikes = spike_df[spike_df[group_by] == pop_name]

        # Filter by time windows
        pop_spikes_1 = pop_spikes[
            (pop_spikes["timestamps"] >= time_window_1[0])
            & (pop_spikes["timestamps"] <= time_window_1[1])
        ]
        pop_spikes_2 = pop_spikes[
            (pop_spikes["timestamps"] >= time_window_2[0])
            & (pop_spikes["timestamps"] <= time_window_2[1])
        ]

        # Get unique neuron IDs
        unique_neurons = pop_spikes["node_ids"].unique()

        # Calculate firing rates per neuron for each time window in Hz
        neuron_rates_1 = []
        neuron_rates_2 = []

        for neuron in unique_neurons:
            # Count spikes for this neuron in each window
            n_spikes_1 = len(pop_spikes_1[pop_spikes_1["node_ids"] == neuron])
            n_spikes_2 = len(pop_spikes_2[pop_spikes_2["node_ids"] == neuron])

            # Calculate firing rate in Hz (convert ms to seconds by dividing by 1000)
            rate_1 = n_spikes_1 / ((time_window_1[1] - time_window_1[0]) / 1000)
            rate_2 = n_spikes_2 / ((time_window_2[1] - time_window_2[0]) / 1000)

            neuron_rates_1.append(rate_1)
            neuron_rates_2.append(rate_2)

        # Calculate average firing rates
        avg_firing_rate_1 = np.mean(neuron_rates_1) if neuron_rates_1 else 0
        avg_firing_rate_2 = np.mean(neuron_rates_2) if neuron_rates_2 else 0

        # Perform Mann-Whitney U test
        # Handle the case when one or both arrays are empty
        if len(neuron_rates_1) > 0 and len(neuron_rates_2) > 0:
            u_stat, p_val = mannwhitneyu(neuron_rates_1, neuron_rates_2, alternative="two-sided")
        else:
            u_stat, p_val = np.nan, np.nan

        print(f"    Average firing rate in window 1: {avg_firing_rate_1:.2f} Hz")
        print(f"    Average firing rate in window 2: {avg_firing_rate_2:.2f} Hz")
        print(f"    U-statistic: {u_stat:.2f}")
        print(f"    p-value: {p_val}")
        print(f"    Significant difference (p<0.05): {'Yes' if p_val < 0.05 else 'No'}")
    return


def find_bursting_cells(
    df: pd.DataFrame,
    isi_threshold: float = 10.0, # Made float explicit
    burst_count_threshold: int = 1,
) -> pd.DataFrame:
    """
    Finds bursting cells in a population based on an Inter-Spike Interval (ISI) threshold.
    Cells identified as bursters will have "_bursters" appended to their 'pop_name'.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing spike data with at least 'timestamps', 'node_ids', and 'pop_name' columns.
    isi_threshold : float, optional
        Time difference threshold in milliseconds (ms) to identify spikes belonging to a burst.
        An ISI less than this threshold is considered part of a burst (default is 10.0 ms).
    burst_count_threshold : int, optional
        Minimum number of short ISIs (burst instances) required to classify a cell as a burster (default is 1).

    Returns
    -------
    pd.DataFrame
        A new DataFrame derived from the input, where the 'pop_name' of bursting cells
        has been appended with "_bursters". Includes original spike data for these cells.
    """
    # Create a new DataFrame with the time differences
    diff_df = df.copy()
    diff_df["time_diff"] = df.groupby("node_ids")["timestamps"].diff()

    # Create a column indicating whether each time difference is a burst
    diff_df["is_burst_instance"] = diff_df["time_diff"] < isi_threshold

    # Group by node_ids and check if any row has a burst instance
    # check if there are enough bursts
    burst_summary = diff_df.groupby("node_ids")["is_burst_instance"].sum() >= burst_count_threshold

    # Convert to a DataFrame with reset index
    burst_cells = burst_summary.reset_index(name="is_burst")

    # merge with original df to get timestamps
    burst_cells = pd.merge(burst_cells, df, on="node_ids")

    # Create a mask for burst cells that don't already have "_bursters" in their name
    burst_mask = burst_cells["is_burst"] & ~burst_cells["pop_name"].str.contains(
        "_bursters", na=False
    )

    # Add "_bursters" suffix only to those cells
    burst_cells.loc[burst_mask, "pop_name"] = burst_cells.loc[burst_mask, "pop_name"] + "_bursters"

    for pop in sorted(burst_cells["pop_name"].unique()):
        print(
            f"Number of cells in {pop}: {burst_cells[burst_cells['pop_name'] == pop]['node_ids'].nunique()}"
        )

    return burst_cells


def find_highest_firing_cells(
    df: pd.DataFrame, upper_quantile: float, groupby: str = "pop_name"
) -> pd.DataFrame:
    """
    Identifies and returns spikes from cells with firing rates above a specified upper quantile,
    grouped by a population label.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing spike data with at least 'timestamps', 'node_ids',
        and the column specified by `groupby` (e.g., 'pop_name').
    upper_quantile : float
        The upper quantile threshold (between 0.0 and 1.0).
        Cells with firing rates in the top (1 - `upper_quantile`) fraction are selected.
        For example, `upper_quantile=0.8` selects the top 20% of high-firing cells.
    groupby : str, optional
        The column name used to group neurons by population (default is 'pop_name').

    Returns
    -------
    pd.DataFrame
        A DataFrame containing only the spikes from the high-firing cells,
        concatenated across all specified groups.
    """
    df_list = []
    for pop in df[groupby].unique():
        pop_df = df[df[groupby] == pop]
        _, pop_fr = compute_firing_rate_stats(pop_df, groupby=groupby)

        # Identify high firing cells
        threshold = pop_fr["firing_rate"].quantile(upper_quantile)
        high_firing_cells = pop_fr[pop_fr["firing_rate"] >= threshold]["node_ids"]

        # Filter spikes for high firing cells
        pop_spikes = pop_df[pop_df["node_ids"].isin(high_firing_cells)]
        df_list.append(pop_spikes)

    # Combine all high firing spikes into one DataFrame
    result_df = pd.concat(df_list, ignore_index=True)
    return result_df
