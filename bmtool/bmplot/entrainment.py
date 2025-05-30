from typing import List, Tuple, Union, Optional, Dict, Any # Added Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib.gridspec import GridSpec
from scipy import stats

from bmtool.analysis import entrainment as bmentr
from bmtool.analysis import spikes as bmspikes
from bmtool.analysis.lfp import get_lfp_power
from bmtool.bmplot.connections import is_notebook # Added for consistent plot showing


def plot_spike_power_correlation(
    spike_rate: xr.DataArray,
    lfp_data: xr.DataArray,
    fs: float,
    pop_names: List[str],
    filter_method: str = "wavelet",
    bandwidth: float = 2.0,
    lowcut: Optional[float] = None,
    highcut: Optional[float] = None,
    freq_range: Tuple[float, float] = (10, 100),
    freq_step: float = 5.0,
    type_name: str = "raw",
    time_windows: Optional[List[Tuple[float, float]]] = None,
    confidence_level: float = 0.95,
    save_path: Optional[str] = None, # Added save_path
) -> None:
    """
    Calculate and plot correlation between population spike rates and LFP power across frequencies.
    Supports both single-signal and trial-based analysis with confidence intervals.

    Parameters
    ----------
    spike_rate : xr.DataArray
        Population spike rates with dimensions (time, population[, type]).
    lfp_data : xr.DataArray
        LFP data, typically with a 'time' dimension.
    fs : float
        Sampling frequency of the LFP data in Hz.
    pop_names : List[str]
        List of population names to analyze from `spike_rate`.
    filter_method : str, optional
        Filtering method for LFP power: 'wavelet' or 'butter'. Default is 'wavelet'.
    bandwidth : float, optional
        Bandwidth parameter for wavelet filter if `filter_method` is 'wavelet'. Default is 2.0.
    lowcut : Optional[float], optional
        Lower frequency bound (Hz) for Butterworth bandpass filter if `filter_method` is 'butter'. Default is None.
    highcut : Optional[float], optional
        Upper frequency bound (Hz) for Butterworth bandpass filter if `filter_method` is 'butter'. Default is None.
    freq_range : Tuple[float, float], optional
        Min and max frequency (Hz) to analyze. Default is (10, 100).
    freq_step : float, optional
        Step size for frequency analysis in Hz. Default is 5.0.
    type_name : str, optional
        Which type of spike rate to use if 'type' dimension exists in `spike_rate` (e.g., 'raw', 'smoothed'). Default is 'raw'.
    time_windows : Optional[List[Tuple[float, float]]], optional
        List of (start_ms, end_ms) time tuples for trial-based analysis. 
        If None, the entire signal duration is analyzed as a single trial. Default is None.
    confidence_level : float, optional
        Confidence level for confidence interval calculation in trial-based analysis. Default is 0.95.
    save_path : Optional[str], optional
        Full path to save the figure. If None, figure is displayed. Default is None.
    """
    # Setup
    frequencies = np.arange(freq_range[0], freq_range[1] + freq_step, freq_step) # Ensure endpoint is included if step aligns
    is_trial_based = time_windows is not None

    power_by_freq: Dict[float, xr.DataArray] = {}
    for freq_val in frequencies:
        power_by_freq[freq_val] = get_lfp_power(
            lfp_data, freq=freq_val, fs=fs, filter_method=filter_method, 
            lowcut=lowcut, highcut=highcut, bandwidth=bandwidth
        )

    results: Dict[str, Dict[float, Dict[str, Any]]] = {}
    for pop in pop_names:
        pop_spike_rate_da = spike_rate.sel(population=pop, type=type_name)
        results[pop] = {}

        for freq_val in frequencies:
            lfp_power_da = power_by_freq[freq_val]

            if not is_trial_based:
                if len(pop_spike_rate_da.time) != len(lfp_power_da.time):
                    print(f"Warning: Length mismatch for {pop} at {freq_val} Hz. Spike rate: {len(pop_spike_rate_da.time)}, LFP power: {len(lfp_power_da.time)}")
                    results[pop][freq_val] = {"correlation": np.nan, "p_value": np.nan}
                    continue
                if len(pop_spike_rate_da.time) < 2: # Spearman R needs at least 2 points
                    results[pop][freq_val] = {"correlation": np.nan, "p_value": np.nan}
                    continue
                corr, p_val = stats.spearmanr(pop_spike_rate_da.data, lfp_power_da.data)
                results[pop][freq_val] = {"correlation": corr, "p_value": p_val}
            else: # Trial-based
                trial_correlations: List[float] = []
                for start_time, end_time in time_windows: # type: ignore # time_windows is checked by is_trial_based
                    trial_spike_rate = pop_spike_rate_da.sel(time=slice(start_time, end_time))
                    trial_lfp_power = lfp_power_da.sel(time=slice(start_time, end_time))

                    if len(trial_spike_rate.time) < 2 or len(trial_lfp_power.time) < 2 or \
                       len(trial_spike_rate.time) != len(trial_lfp_power.time):
                        continue
                    
                    corr, _ = stats.spearmanr(trial_spike_rate.data, trial_lfp_power.data)
                    if not np.isnan(corr):
                        trial_correlations.append(corr)
                
                if trial_correlations:
                    trial_corrs_arr = np.array(trial_correlations)
                    mean_corr = np.mean(trial_corrs_arr)
                    ci_lower, ci_upper = mean_corr, mean_corr
                    if len(trial_corrs_arr) > 1:
                        sem = stats.sem(trial_corrs_arr)
                        if sem > 0:
                            alpha = 1.0 - confidence_level
                            df = len(trial_corrs_arr) - 1
                            t_crit = stats.t.ppf(1.0 - alpha / 2.0, df)
                            ci_lower = mean_corr - t_crit * sem
                            ci_upper = mean_corr + t_crit * sem
                    results[pop][freq_val] = {
                        "correlation": mean_corr, "ci_lower": ci_lower, "ci_upper": ci_upper,
                        "n_trials": len(trial_corrs_arr), "trial_correlations": trial_corrs_arr
                    }
                else:
                    results[pop][freq_val] = {
                        "correlation": np.nan, "ci_lower": np.nan, "ci_upper": np.nan,
                        "n_trials": 0, "trial_correlations": np.array([])
                    }
    # Plotting
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 8))

    for i, pop in enumerate(pop_names):
        plot_freqs, plot_corrs, plot_ci_lower, plot_ci_upper = [], [], [], []
        for freq_val in frequencies:
            if freq_val in results[pop] and not np.isnan(results[pop][freq_val]["correlation"]):
                plot_freqs.append(freq_val)
                plot_corrs.append(results[pop][freq_val]["correlation"])
                if is_trial_based:
                    plot_ci_lower.append(results[pop][freq_val]["ci_lower"])
                    plot_ci_upper.append(results[pop][freq_val]["ci_upper"])
        
        if not plot_freqs: continue
        color = plt.cm.get_cmap('tab10', len(pop_names))(i) # Use get_cmap
        ax.plot(plot_freqs, plot_corrs, marker="o", label=pop, linewidth=2, markersize=6, color=color)
        if is_trial_based and plot_ci_lower:
            ax.fill_between(plot_freqs, plot_ci_lower, plot_ci_upper, alpha=0.2, color=color)

    ax.set_xlabel("Frequency (Hz)", fontsize=12)
    ylabel = "Mean Spike Rate-Power Correlation" if is_trial_based else "Spike Rate-Power Correlation"
    ax.set_ylabel(ylabel, fontsize=12)
    plot_title = f"{'Trial-averaged ' if is_trial_based else ''}Spike Rate-LFP Power Correlation"
    ax.set_title(plot_title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.5)

    legend_elements = [plt.Line2D([0], [0], color=plt.cm.get_cmap('tab10', len(pop_names))(i), marker="o", linestyle="-", label=pop) for i, pop in enumerate(pop_names)] # Use get_cmap
    if is_trial_based:
        legend_elements.append(plt.Line2D([0], [0], color="gray", alpha=0.3, linewidth=10, label=f"{int(confidence_level*100)}% CI"))
    ax.legend(handles=legend_elements, fontsize=10, loc="best")

    if len(frequencies) > 10:
        ax.set_xticks(frequencies[::2])
    else:
        ax.set_xticks(frequencies)
    if frequencies.size > 0:
        ax.set_xlim(frequencies[0], frequencies[-1])
    
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(min(y_min, -0.1), max(y_max, 0.1))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    
    if not is_notebook() and not save_path:
        plt.show()
    elif is_notebook() and not save_path:
        plt.show()
    elif save_path: # If saved, close the plot to free memory, common in batch processing
        plt.close(fig)


def plot_cycle_with_spike_histograms(
    phase_data: Dict[str, np.ndarray], 
    bins: int = 36, 
    pop_names_to_plot: Optional[List[str]] = None,
    save_path: Optional[str] = None, # Added save_path
) -> None:
    """
    Plot an idealized cycle with spike histograms for different neuron populations.

    Parameters
    ----------
    phase_data : Dict[str, np.ndarray]
        Dictionary where keys are population names and values are numpy arrays
        of spike phases (in radians, from -pi to pi) for that population.
    bins : int, optional
        Number of bins for the phase histogram (default is 36, giving 10-degree bins).
    pop_names_to_plot : Optional[List[str]], optional
        List of population names to be plotted. If None, all populations in `phase_data` are plotted.
        Default is None.
    save_path : Optional[str], optional
        Full path to save the figure. If None, figure is displayed. Default is None.
    """
    sns.set_style("whitegrid")

    if pop_names_to_plot is None:
        pop_names_to_plot = list(phase_data.keys())
    
    if not pop_names_to_plot:
        print("No populations specified or found in phase_data. Nothing to plot.")
        return

    fig = plt.figure(figsize=(12, 2 * (len(pop_names_to_plot) + 1.5))) # Adjusted figsize
    gs = GridSpec(len(pop_names_to_plot) + 1, 1, height_ratios=[1.5] + [1] * len(pop_names_to_plot))

    ax_gamma = fig.add_subplot(gs[0])
    x_cycle = np.linspace(-np.pi, np.pi, 1000)
    y_cycle = np.sin(x_cycle)
    ax_gamma.plot(x_cycle, y_cycle, "b-", linewidth=2)
    ax_gamma.set_title("Cycle with Neuron Population Spike Distributions", fontsize=14)
    ax_gamma.set_ylabel("Amplitude", fontsize=12)
    ax_gamma.set_xlim(-np.pi, np.pi)
    ax_gamma.set_xticks(np.linspace(-np.pi, np.pi, 9))
    ax_gamma.set_xticklabels(["-180°", "-135°", "-90°", "-45°", "0°", "45°", "90°", "135°", "180°"])
    ax_gamma.grid(True)
    ax_gamma.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    ax_gamma.axvline(x=0, color="k", linestyle="--", alpha=0.3)

    pop_colors = plt.cm.get_cmap('tab10', len(pop_names_to_plot)) # Use get_cmap

    for i, pop_name_str in enumerate(pop_names_to_plot): # Renamed pop_name
        if pop_name_str not in phase_data:
            print(f"Warning: Population '{pop_name_str}' not found in phase_data. Skipping.")
            continue
        
        ax_hist = fig.add_subplot(gs[i + 1], sharex=ax_gamma)
        current_phase_data = phase_data[pop_name_str]
        
        if current_phase_data.size == 0:
            print(f"Warning: No phase data for population '{pop_name_str}'. Plotting empty histogram.")
            hist_values = np.zeros(bins) # Ensure hist_values is defined
        else:
            hist_values, bin_edges = np.histogram(current_phase_data, bins=bins, range=(-np.pi, np.pi))
        
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        if np.sum(hist_values) > 0:
            hist_values = hist_values / np.sum(hist_values) * 100  # Convert to percentage
        
        ax_hist.bar(bin_centers, hist_values, width=(2 * np.pi / bins) * 0.9, alpha=0.7, color=pop_colors(i)) # Added 0.9 width factor
        ax_hist.set_ylabel(f"{pop_name_str}\nSpikes (%)", fontsize=10)
        ax_hist.grid(True, alpha=0.3)
        ax_hist.set_ylim(0, max(hist_values.max() * 1.2, 1)) # Ensure ylim is at least 1 for empty hists

    # Set x-label for the last subplot (if any histograms were plotted)
    if len(pop_names_to_plot) > 0 : # Check if any pop was processed
        fig.get_axes()[-1].set_xlabel("Phase (degrees)", fontsize=12)


    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    if not is_notebook() and not save_path:
        plt.show()
    elif is_notebook() and not save_path:
        plt.show()
    elif save_path:
        plt.close(fig)


def plot_entrainment_by_population(
    ppc_dict: Dict[str, Dict[str, Dict[float, float]]], 
    pop_names: List[str], 
    freqs: Union[List[float], np.ndarray], 
    figsize: Tuple[float, float] = (15, 8), 
    title: Optional[str] = None,
    save_path: Optional[str] = None, # Added save_path
    y_label: str = "PPC Value" # Added y_label for flexibility (PPC, PLV, etc.)
) -> None:
    """
    Plot entrainment metric (PPC, PLV, etc.) for all node populations on one graph 
    with mean and standard error bars.

    Parameters
    ----------
    ppc_dict : Dict[str, Dict[str, Dict[float, float]]]
        Dictionary containing entrainment data. Structure: 
        {pop_name: {node_id: {frequency: value}}}.
    pop_names : List[str]
        List of population names to plot data for.
    freqs : Union[List[float], np.ndarray]
        List or array of frequencies to plot on the x-axis.
    figsize : Tuple[float, float], optional
        Figure size for the plot. Default is (15, 8).
    title : Optional[str], optional
        Title for the plot. If None, a generic title is used. Default is None.
    save_path : Optional[str], optional
        Full path to save the figure. If None, figure is displayed. Default is None.
    y_label : str, optional
        Label for the y-axis (e.g., "PPC Value", "PLV Value"). Default is "PPC Value".
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=figsize) # Use ax for plotting

    n_groups = len(freqs)
    n_populations = len(pop_names)
    group_width = 0.8 
    bar_width = group_width / n_populations if n_populations > 0 else group_width

    pop_colors_map = plt.cm.get_cmap('tab10', n_populations if n_populations > 0 else 1) # Use get_cmap

    x_centers = np.arange(n_groups)
    tick_labels = [f"{freq:.1f}" for freq in freqs] # Format frequency labels

    for i, pop in enumerate(pop_names):
        means, errors, valid_freqs_indices = [], [], []
        if pop not in ppc_dict:
            print(f"Warning: Population '{pop}' not found in ppc_dict. Skipping.")
            continue

        for freq_idx, freq in enumerate(freqs):
            freq_values = [
                ppc_dict[pop][node][freq]
                for node in ppc_dict[pop]
                if freq in ppc_dict[pop][node] and ppc_dict[pop][node][freq] is not None and not np.isnan(ppc_dict[pop][node][freq])
            ]
            
            if freq_values:
                means.append(np.mean(freq_values))
                errors.append(stats.sem(freq_values) if len(freq_values) > 1 else 0)
                valid_freqs_indices.append(freq_idx)
        
        if not means: # Skip if no valid data for this population
            continue

        current_x_centers = x_centers[valid_freqs_indices]
        x_positions = current_x_centers + (i - n_populations / 2 + 0.5) * bar_width
        
        ax.bar(
            x_positions, means, width=bar_width * 0.9, color=pop_colors_map(i), alpha=0.7, label=pop
        )
        ax.errorbar(x_positions, means, yerr=errors, fmt="none", ecolor="black", capsize=4)

    ax.set_xlabel("Frequency (Hz)", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    plot_title = title if title else f"{y_label} by Population and Frequency"
    ax.set_title(plot_title, fontsize=14)
    ax.set_xticks(x_centers)
    ax.set_xticklabels(tick_labels)
    ax.legend(title="Population", bbox_to_anchor=(1.05, 1), loc='upper left') # Adjust legend

    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for external legend
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    if not is_notebook() and not save_path:
        plt.show()
    elif is_notebook() and not save_path:
        plt.show()
    elif save_path:
        plt.close(fig)


def plot_entrainment_swarm_plot(
    entrainment_data: Dict[str, Dict[str, Dict[float, float]]], # Renamed ppc_dict
    pop_names: List[str], 
    freq: float, 
    save_path: Optional[str] = None, 
    title: Optional[str] = None,
    y_label: str = "Entrainment Value", # Added y_label
    palette: str = "Set2" # Added palette
) -> Optional[plt.Figure]:
    """
    Plot a swarm plot of the entrainment metric for different populations at a single frequency.

    Parameters
    ----------
    entrainment_data : Dict[str, Dict[str, Dict[float, float]]]
        Dictionary containing entrainment values (PPC, PLV, etc.).
        Structure: {pop_name: {node_id: {frequency: value}}}.
    pop_names : List[str]
        List of population names to include in the plot.
    freq : float
        The specific frequency (Hz) to plot.
    save_path : Optional[str], optional
        Full path to save the figure (e.g., 'path/to/figure.png'). 
        If None, figure is displayed. Default is None.
    title : Optional[str], optional
        Title for the plot. If None, a default title is generated. Default is None.
    y_label : str, optional
        Label for the y-axis. Default is "Entrainment Value".
    palette : str, optional
        Color palette for seaborn swarmplot. Default is "Set2".

    Returns
    -------
    Optional[matplotlib.figure.Figure]
        The figure object if data was plotted, otherwise None.
    """
    sns.set_style("whitegrid")
    data_list: List[Dict[str, Any]] = []

    for pop in pop_names:
        if pop not in entrainment_data:
            print(f"Warning: Population {pop} not in entrainment_data. Skipping.")
            continue
        for node in entrainment_data[pop]:
            if freq in entrainment_data[pop][node] and entrainment_data[pop][node][freq] is not None \
               and not np.isnan(entrainment_data[pop][node][freq]):
                data_list.append(
                    {"Population": pop, "Node": node, y_label: entrainment_data[pop][node][freq]}
                )

    if not data_list:
        print(f"No data available for frequency {freq} Hz across specified populations.")
        return None

    df = pd.DataFrame(data_list)

    print(f"\nSummary statistics at {freq} Hz ({y_label}):")
    for pop in pop_names:
        subset = df[df["Population"] == pop]
        if not subset.empty:
            mean_val = subset[y_label].mean()
            n = len(subset)
            sem_val = subset[y_label].sem() if n > 1 else 0.0
            print(f"  {pop}: {mean_val:.4f} ± {sem_val:.4f} (n={n})")

    fig, ax = plt.subplots(figsize=(max(8, len(pop_names) * 1.5), 8))
    sns.swarmplot(x="Population", y=y_label, data=df, size=3, ax=ax, palette=palette)

    y_min_data, y_max_data = df[y_label].min(), df[y_label].max()
    y_plot_range = y_max_data - y_min_data
    if y_plot_range == 0: y_plot_range = 1.0 # Avoid division by zero if all values are same

    for i, pop in enumerate(pop_names):
        subset = df[df["Population"] == pop]
        if not subset.empty:
            n = len(subset)
            # Position 'n' annotation slightly below the swarm points for that category
            # This requires finding min y for current category, or placing it at a fixed offset from axis
            pop_y_min = subset[y_label].min()
            ax.annotate(f"n={n}", (i, pop_y_min - 0.05 * y_plot_range - 0.05), ha="center", fontsize=10)
            
            mean_val = subset[y_label].mean()
            ax.plot([i - 0.25, i + 0.25], [mean_val, mean_val], "r-", linewidth=2)

    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.7)

    if len(pop_names) > 1:
        print(f"\nMann-Whitney U Test Results at {freq} Hz ({y_label}):")
        print("-" * 60)
        y_max_plot = y_max_data # Start with data max for annotation positioning

        for i in range(len(pop_names)):
            for j in range(i + 1, len(pop_names)):
                pop1, pop2 = pop_names[i], pop_names[j]
                vals1 = df[df["Population"] == pop1][y_label].dropna().values
                vals2 = df[df["Population"] == pop2][y_label].dropna().values

                if len(vals1) > 1 and len(vals2) > 1:
                    u_stat, p_val = stats.mannwhitneyu(vals1, vals2, alternative="two-sided")
                    sig_str = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                    
                    bar_height = y_max_plot + 0.05 * y_plot_range * (1 + (j - i -1)*0.2) # Adjust vertical spacing for multiple bars
                    ax.plot([i, j], [bar_height, bar_height], "k-", lw=1)
                    ax.plot([i, i], [bar_height - 0.01 * y_plot_range, bar_height], "k-", lw=1)
                    ax.plot([j, j], [bar_height - 0.01 * y_plot_range, bar_height], "k-", lw=1)
                    ax.text((i + j) / 2, bar_height + 0.005 * y_plot_range, sig_str, ha="center", va="bottom", fontsize=10)
                    y_max_plot = bar_height # Update max y for next annotation bar
                    print(f"  {pop1} vs {pop2}: U={u_stat:.1f}, p={p_val:.4f} ({sig_str})")

    ax.set_xlabel("Population", fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    plot_title = title if title else f"Population Entrainment ({y_label}) at {freq} Hz"
    ax.set_title(plot_title, fontsize=16)
    
    current_y_min, current_y_max = ax.get_ylim()
    ax.set_ylim(min(current_y_min, y_min_data - 0.15 * y_plot_range), max(current_y_max, y_max_plot + 0.05 * y_plot_range))


    ax.grid(True, linestyle="--", alpha=0.7, axis="y")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    
    if not is_notebook() and not save_path:
        plt.show()
    elif is_notebook() and not save_path:
        plt.show()
    elif save_path:
        plt.close(fig)
        
    return fig


def plot_trial_avg_entrainment(
    spike_df: pd.DataFrame,
    lfp_data: xr.DataArray, # Changed from np.ndarray to xr.DataArray
    time_windows: List[Tuple[float, float]],
    entrainment_method: str,
    pop_names: List[str],
    freqs: Union[List[float], np.ndarray],
    firing_quantile: float,
    spike_fs: float = 1000.0, # Made float explicit
    lfp_fs: Optional[float] = None, # Added lfp_fs, as it's needed if lfp_data doesn't have fs attr
    error_type: str = "ci",  
    save_path: Optional[str] = None, # Added save_path
) -> None:
    """
    Plot trial-averaged entrainment for specified population names. 
    Currently assumes wavelet filter for phase/power estimation.

    Parameters
    ----------
    spike_df : pd.DataFrame
        Spike data with 'timestamps', 'node_ids', and 'pop_name' columns.
    lfp_data : xr.DataArray
        LFP data as an xarray DataArray. Must have a time coordinate and an 'fs' attribute
        if `lfp_fs` parameter is not provided.
    time_windows : List[Tuple[float, float]]
        List of (start_ms, end_ms) tuples for each trial.
    entrainment_method : str
        Method for entrainment: 'ppc', 'ppc2', or 'plv'.
    pop_names : List[str]
        List of population names to process.
    freqs : Union[List[float], np.ndarray]
        Array of frequencies (Hz) to analyze.
    firing_quantile : float
        Upper quantile (0-1) for selecting high-firing cells (e.g., 0.8 for top 20%).
    spike_fs : float, optional
        Sampling frequency (Hz) for `spike_df`. Default is 1000.0.
    lfp_fs : Optional[float], optional
        Sampling frequency (Hz) for `lfp_data`. If None, attempts to get from `lfp_data.attrs['fs']`. Default is None.
    error_type : str, optional
        Error bars: "ci" (95% confidence interval), "sem" (standard error), "std" (standard deviation). Default is "ci".
    save_path : Optional[str], optional
        Full path to save the figure. If None, figure is displayed. Default is None.

    Raises
    ------
    ValueError
        If arguments are invalid or LFP sampling frequency cannot be determined.
    """
    sns.set_style("whitegrid")
    if entrainment_method not in ["ppc", "plv", "ppc2"]:
        raise ValueError("entrainment_method must be 'ppc', 'ppc2', or 'plv'")
    if error_type not in ["ci", "sem", "std"]:
        raise ValueError("error_type must be 'ci', 'sem', or 'std'")
    if not (0 < firing_quantile < 1):
        raise ValueError("firing_quantile must be between 0.0 and 1.0")

    _lfp_fs: float
    if lfp_fs is not None:
        _lfp_fs = lfp_fs
    elif 'fs' in lfp_data.attrs:
        _lfp_fs = lfp_data.attrs['fs']
    else:
        raise ValueError("LFP sampling frequency `lfp_fs` must be provided or be an attribute 'fs' of `lfp_data`.")

    freqs_arr = np.array(freqs)
    all_entrainment_data: Dict[str, List[List[float]]] = {pop: [] for pop in pop_names}

    for trial_idx, (t_start_trial, t_stop_trial) in enumerate(time_windows):
        trial_spikes_df = spike_df[
            (spike_df["timestamps"] >= t_start_trial) & (spike_df["timestamps"] <= t_stop_trial)
        ].copy()

        pop_spike_data_for_trial: Dict[str, pd.DataFrame] = {}
        for pop_name in pop_names:
            pop_trial_spikes = trial_spikes_df[trial_spikes_df["pop_name"] == pop_name]
            if pop_trial_spikes.empty:
                # print(f"Warning: No spikes for {pop_name} in trial {trial_idx}.") # Optional: too verbose
                all_entrainment_data[pop_name].append([np.nan] * len(freqs_arr))
                continue 
            
            # Consider if find_highest_firing_cells should be applied per trial or across all data once
            # Applying per trial as in original logic:
            high_firing_pop_spikes = bmspikes.find_highest_firing_cells(
                pop_trial_spikes, upper_quantile=firing_quantile, groupby='pop_name' # Ensure groupby is passed
            )
            if high_firing_pop_spikes.empty:
                # print(f"Warning: No high-firing spikes for {pop_name} in trial {trial_idx} after quantile filter.") # Optional
                all_entrainment_data[pop_name].append([np.nan] * len(freqs_arr))
                continue
            pop_spike_data_for_trial[pop_name] = high_firing_pop_spikes
        
        # Pre-filter LFP for the current trial window to pass to entrainment functions
        lfp_trial_data = lfp_data.sel(time=slice(t_start_trial, t_stop_trial))
        if lfp_trial_data.time.size == 0:
            print(f"Warning: LFP data is empty for trial {trial_idx}. Skipping entrainment calculation for this trial.")
            for pop_name in pop_names: # Ensure lists remain consistent in length
                 if pop_name in pop_spike_data_for_trial: # Only if it was not skipped due to no spikes
                    all_entrainment_data[pop_name].append([np.nan] * len(freqs_arr))
            continue


        trial_pop_entrainment_values: Dict[str, List[float]] = {pop: [] for pop in pop_names if pop in pop_spike_data_for_trial}
        for freq_val in freqs_arr:
            for pop_name, current_pop_spikes in pop_spike_data_for_trial.items():
                try:
                    val = np.nan
                    if entrainment_method == "ppc":
                        val = bmentr.calculate_ppc(current_pop_spikes["timestamps"].values, lfp_trial_data, spike_fs=spike_fs, lfp_fs=_lfp_fs, freq=freq_val, filter_method="wavelet", ppc_method="numpy") # Changed to numpy for less deps
                    elif entrainment_method == "plv":
                        val = bmentr.calculate_spike_lfp_plv(current_pop_spikes["timestamps"].values, lfp_trial_data, spike_fs=spike_fs, lfp_fs=_lfp_fs, freq=freq_val, filter_method="wavelet")
                    elif entrainment_method == "ppc2":
                        val = bmentr.calculate_ppc2(current_pop_spikes["timestamps"].values, lfp_trial_data, spike_fs=spike_fs, lfp_fs=_lfp_fs, freq=freq_val, filter_method="wavelet")
                    trial_pop_entrainment_values[pop_name].append(val)
                except Exception as e:
                    # print(f"Warning: Error calculating {entrainment_method} for {pop_name} at {freq_val}Hz in trial {trial_idx}: {e}") # Optional
                    trial_pop_entrainment_values[pop_name].append(np.nan)
        
        for pop_name in pop_names:
            if pop_name in trial_pop_entrainment_values:
                 all_entrainment_data[pop_name].append(trial_pop_entrainment_values[pop_name])
            elif pop_name not in pop_spike_data_for_trial: # Was skipped due to no spikes initially
                pass # Already handled by appending NaNs earlier or will be handled by nanmean

    mean_entrainment: Dict[str, np.ndarray] = {}
    error_bars: Dict[str, np.ndarray] = {}

    for pop_name in pop_names:
        # Ensure all sublists have same length (len(freqs_arr)), padding with NaNs if necessary from earlier skips
        max_len = len(freqs_arr)
        padded_data = [lst if len(lst) == max_len else [np.nan]*max_len for lst in all_entrainment_data[pop_name]]
        pop_data_arr = np.array(padded_data, dtype=float)

        with np.errstate(invalid="ignore", divide="ignore"): # Handles all-NaN slices
            mean_entrainment[pop_name] = np.nanmean(pop_data_arr, axis=0)
            valid_counts = np.sum(~np.isnan(pop_data_arr), axis=0)
            std_dev = np.nanstd(pop_data_arr, axis=0, ddof=1)

            if error_type == "ci":
                sem_val = std_dev / np.sqrt(valid_counts)
                # Replace inf with nan where valid_counts is 0
                sem_val[valid_counts == 0] = np.nan 
                error_val = np.full_like(sem_val, np.nan)
                # Calculate t_value only where valid_counts > 1
                # Using a default large t_value (approx for 95% CI) if too few trials, or more precise if possible
                for i_freq in range(len(freqs_arr)):
                    if valid_counts[i_freq] > 1:
                         t_val = stats.t.ppf(0.975, valid_counts[i_freq] - 1)
                         error_val[i_freq] = t_val * sem_val[i_freq]
                    elif valid_counts[i_freq] == 1: # Cannot compute CI for 1 sample, error is effectively infinite or undefined
                         error_val[i_freq] = np.nan # Or some indicator like mean itself if you want to show the point
            elif error_type == "sem":
                error_val = std_dev / np.sqrt(valid_counts)
                error_val[valid_counts == 0] = np.nan
            elif error_type == "std":
                error_val = std_dev
            error_bars[pop_name] = error_val
            
    fig, ax = plt.subplots(figsize=(12, 8))
    markers = ["o-", "s-", "^-", "D-", "v-", "p-", "*-", "h-"]
    pop_colors_map = plt.cm.get_cmap('tab10', len(pop_names) if len(pop_names) > 0 else 1)

    for i, pop_name in enumerate(pop_names):
        marker = markers[i % len(markers)]
        color = pop_colors_map(i)
        valid_mask = ~np.isnan(mean_entrainment[pop_name])
        
        if np.any(valid_mask):
            ax.plot(
                freqs_arr[valid_mask], mean_entrainment[pop_name][valid_mask], marker,
                linewidth=2, label=pop_name, color=color, markersize=6
            )
            if pop_name in error_bars and not np.all(np.isnan(error_bars[pop_name])):
                lower_bound = (mean_entrainment[pop_name] - error_bars[pop_name])[valid_mask]
                upper_bound = (mean_entrainment[pop_name] + error_bars[pop_name])[valid_mask]
                ax.fill_between(freqs_arr[valid_mask], lower_bound, upper_bound, alpha=0.2, color=color)

    ax.set_xlabel("Frequency (Hz)", fontsize=12)
    ax.set_ylabel(f"{entrainment_method.upper()}", fontsize=12)
    firing_percentage = int((1 - firing_quantile) * 100)
    error_labels = {"ci": "95% CI", "sem": "±SEM", "std": "±SD"}
    error_label = error_labels.get(error_type, error_type) # Get label or use error_type itself
    
    plot_title = f"Trial-Averaged {entrainment_method.upper()} for Top {firing_percentage}% Firing Cells ({error_label})"
    ax.set_title(plot_title, fontsize=14)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    if not is_notebook() and not save_path:
        plt.show()
    elif is_notebook() and not save_path:
        plt.show()
    elif save_path:
        plt.close(fig)
