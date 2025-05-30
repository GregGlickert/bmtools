"""Plotting functions for neural spikes and firing rates."""

from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from ..util.util import load_nodes_from_config


def raster(
    spikes_df: Optional[pd.DataFrame] = None,
    config: Optional[str] = None,
    network_name: Optional[str] = None,
    groupby: Optional[str] = "pop_name", # Assuming groupby refers to a single column name here
    ax: Optional[Axes] = None,
    tstart: Optional[float] = None,
    tstop: Optional[float] = None,
    color_map: Optional[Dict[str, Any]] = None, # Color can be many things (hex, rgb tuple, name)
    dot_size: float = 0.3, # Made non-optional as it has a default
) -> Axes:
    """
    Plots a raster plot of neural spikes, with different colors for each population group.

    Parameters
    ----------
    spikes_df : pd.DataFrame, optional
        DataFrame containing spike data. Must include 'timestamps' and 'node_ids'.
        If `config` is not provided, this DataFrame must also contain the column specified by `groupby`.
        Default is None (though practically required if `config` not used for population info).
    config : Optional[str], optional
        Path to the BMTK simulation configuration file. If provided, node information
        (including the `groupby` column) will be loaded and merged. Default is None.
    network_name : Optional[str], optional
        Name of the network within the config to use for loading node data.
        If None and `config` is used, attempts to use the first network found. Default is None.
    groupby : Optional[str], optional
        Column name in `spikes_df` (or in node attributes loaded from `config`) to use for
        grouping spikes by color (e.g., 'pop_name'). Default is "pop_name".
    ax : Optional[plt.Axes], optional
        Matplotlib Axes object to plot on. If None, a new figure and axes are created. Default is None.
    tstart : Optional[float], optional
        Start time (ms) for filtering spikes. Spikes before this time are excluded. Default is None.
    tstop : Optional[float], optional
        Stop time (ms) for filtering spikes. Spikes after this time are excluded. Default is None.
    color_map : Optional[Dict[str, Any]], optional
        Dictionary mapping group names (from `groupby` column) to matplotlib-compatible color specifications.
        If None, a default colormap ('tab10') will be used. Default is None.
    dot_size : float, optional
        Size of the dots in the raster plot. Default is 0.3.

    Returns
    -------
    plt.Axes
        The Matplotlib Axes object containing the raster plot.

    Raises
    ------
    ValueError
        If `groupby` column is not found or `color_map` is missing entries for existing groups.
    KeyError
        If specified `network_name` is not found in the configuration.
        
    Notes
    -----
    - If `config` is provided, it's used to fetch node attributes (like 'pop_name') and merge them
      with the `spikes_df` based on 'node_ids'.
    - Ensure the `groupby` column exists either in the initial `spikes_df` or in the node attributes
      loaded via `config`.
    """
    # Initialize axes if none provided
    sns.set_style("whitegrid")
    if ax is None:
        _, ax = plt.subplots(1, 1)

    # Filter spikes by time range if specified
    if tstart is not None:
        spikes_df = spikes_df[spikes_df["timestamps"] > tstart]
    if tstop is not None:
        spikes_df = spikes_df[spikes_df["timestamps"] < tstop]

    # Load and merge node population data if config is provided
    if config:
        nodes = load_nodes_from_config(config)
        if network_name:
            nodes = nodes.get(network_name, {})
        else:
            nodes = list(nodes.values())[0] if nodes else {}
            print(
                "Grabbing first network; specify a network name to ensure correct node population is selected."
            )

        # Find common columns, but exclude the join key from the list
        common_columns = spikes_df.columns.intersection(nodes.columns).tolist()
        common_columns = [
            col for col in common_columns if col != "node_ids"
        ]  # Remove our join key from the common list

        # Drop all intersecting columns except the join key column from df2
        spikes_df = spikes_df.drop(columns=common_columns)
        # merge nodes and spikes df
        spikes_df = spikes_df.merge(
            nodes[groupby], left_on="node_ids", right_index=True, how="left"
        )

    # Get unique population names
    unique_pop_names = spikes_df[groupby].unique()

    # Generate colors if no color_map is provided
    if color_map is None:
        cmap = plt.get_cmap("tab10")  # Default colormap
        color_map = {
            pop_name: cmap(i / len(unique_pop_names)) for i, pop_name in enumerate(unique_pop_names)
        }
    else:
        # Ensure color_map contains all population names
        missing_colors = [pop for pop in unique_pop_names if pop not in color_map]
        if missing_colors:
            raise ValueError(f"color_map is missing colors for populations: {missing_colors}")

    # Plot each population with its specified or generated color
    legend_handles = []
    for pop_name, group in spikes_df.groupby(groupby):
        ax.scatter(group["timestamps"], group["node_ids"], color=color_map[pop_name], s=dot_size)
        # Dummy scatter for consistent legend appearance
        handle = ax.scatter([], [], color=color_map[pop_name], label=pop_name, s=20)
        legend_handles.append(handle)

    # Label axes
    ax.set_xlabel("Time")
    ax.set_ylabel("Node ID")
    ax.legend(handles=legend_handles, title="Population", loc="upper right", framealpha=0.9)

    return ax


# uses df from bmtool.analysis.spikes compute_firing_rate_stats
def plot_firing_rate_pop_stats(
    firing_stats: pd.DataFrame,
    groupby: Union[str, List[str]],
    ax: Optional[Axes] = None,
    color_map: Optional[Dict[str, Any]] = None, # Color can be many things
) -> Axes:
    """
    Plots a bar graph of mean firing rates with error bars representing standard deviation.

    Parameters
    ----------
    firing_stats : pd.DataFrame
        DataFrame containing pre-computed firing rate statistics. Must include
        columns specified by `groupby`, 'firing_rate_mean', and 'firing_rate_std'.
    groupby : Union[str, List[str]]
        Column name(s) in `firing_stats` used to define the groups for bars.
    ax : Optional[plt.Axes], optional
        Matplotlib Axes object to plot on. If None, a new figure and axes are created. Default is None.
    color_map : Optional[Dict[str, Any]], optional
        Dictionary mapping group names (generated from `groupby` columns) to 
        matplotlib-compatible color specifications. If None, a default colormap ('viridis')
        will be used. Default is None.

    Returns
    -------
    plt.Axes
        The Matplotlib Axes object containing the bar plot.
        
    Raises
    ------
    ValueError
        If `color_map` is provided but is missing entries for some groups.
    """
    # Ensure groupby is a list for consistent handling
    sns.set_style("whitegrid")
    if isinstance(groupby, str):
        groupby = [groupby]

    # Create a categorical column for grouping
    firing_stats["group"] = firing_stats[groupby].astype(str).agg("_".join, axis=1)

    # Get unique group names
    unique_groups = firing_stats["group"].unique()

    # Generate colors if no color_map is provided
    if color_map is None:
        cmap = plt.get_cmap("viridis")
        color_map = {group: cmap(i / len(unique_groups)) for i, group in enumerate(unique_groups)}
    else:
        # Ensure color_map contains all groups
        missing_colors = [group for group in unique_groups if group not in color_map]
        if missing_colors:
            raise ValueError(f"color_map is missing colors for groups: {missing_colors}")

    # Create new figure and axes if ax is not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Sort data for consistent plotting
    firing_stats = firing_stats.sort_values(by="group")

    # Extract values for plotting
    x_labels = firing_stats["group"]
    means = firing_stats["firing_rate_mean"]
    std_devs = firing_stats["firing_rate_std"]

    # Get colors for each group
    colors = [color_map[group] for group in x_labels]

    # Create bar plot
    bars = ax.bar(x_labels, means, yerr=std_devs, capsize=5, color=colors, edgecolor="black")

    # Add error bars manually with caps
    _, caps, _ = ax.errorbar(
        x=np.arange(len(x_labels)),
        y=means,
        yerr=std_devs,
        fmt="none",
        capsize=5,
        capthick=2,
        color="black",
    )

    # Formatting
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_xlabel("Population Group")
    ax.set_ylabel("Mean Firing Rate (spikes/s)")
    ax.set_title("Firing Rate Statistics by Population")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    return ax


# uses df from bmtool.analysis.spikes compute_firing_rate_stats
def plot_firing_rate_distribution(
    individual_stats: pd.DataFrame,
    groupby: Union[str, List[str]],
    ax: Optional[Axes] = None,
    color_map: Optional[Dict[str, Any]] = None, # Color can be many things
    plot_type: Union[str, List[str]] = "box",
    swarm_alpha: float = 0.6,
) -> Axes:
    """
    Plots a distribution of individual firing rates using one or more plot types
    (box plot, violin plot, or swarm plot), overlaying them on top of each other.

    Parameters
    ----------
    individual_stats : pd.DataFrame
        DataFrame containing individual firing rates. Must include 'firing_rate' column
        and the column(s) specified by `groupby`.
    groupby : Union[str, List[str]]
        Column name(s) in `individual_stats` used to define groups for plotting.
    ax : Optional[plt.Axes], optional
        Matplotlib Axes object to plot on. If None, a new figure and axes are created. Default is None.
    color_map : Optional[Dict[str, Any]], optional
        Dictionary mapping group names (generated from `groupby` columns) to 
        matplotlib-compatible color specifications. If None, a default colormap ('viridis')
        will be used. Default is None.
    plot_type : Union[str, List[str]], optional
        A single plot type string or a list of plot type strings to generate.
        Options: "box", "violin", "swarm". Default is "box".
    swarm_alpha : float, optional
        Transparency (alpha value) for swarm plot points. Default is 0.6.

    Returns
    -------
    plt.Axes
        The Matplotlib Axes object containing the overlaid distribution plot(s).
        
    Raises
    ------
    ValueError
        If an invalid `plot_type` is specified or if `color_map` is missing entries.
    """
    sns.set_style("whitegrid")
    # Ensure groupby is a list for consistent handling
    if isinstance(groupby, str):
        groupby = [groupby]

    # Create a categorical column for grouping
    individual_stats["group"] = individual_stats[groupby].astype(str).agg("_".join, axis=1)

    # Validate plot_type (it can be a list or a single type)
    if isinstance(plot_type, str):
        plot_type = [plot_type]

    for pt in plot_type:
        if pt not in ["box", "violin", "swarm"]:
            raise ValueError("plot_type must be one of: 'box', 'violin', 'swarm'.")

    # Get unique groups for coloring
    unique_groups = individual_stats["group"].unique()

    # Generate colors if no color_map is provided
    if color_map is None:
        cmap = plt.get_cmap("viridis")
        color_map = {group: cmap(i / len(unique_groups)) for i, group in enumerate(unique_groups)}

    # Ensure color_map contains all groups
    missing_colors = [group for group in unique_groups if group not in color_map]
    if missing_colors:
        raise ValueError(f"color_map is missing colors for groups: {missing_colors}")

    # Create new figure and axes if ax is not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Sort data for consistent plotting
    individual_stats = individual_stats.sort_values(by="group")

    # Loop over each plot type and overlay them
    for pt in plot_type:
        if pt == "box":
            sns.boxplot(
                data=individual_stats,
                x="group",
                y="firing_rate",
                ax=ax,
                palette=color_map,
                width=0.5,
            )
        elif pt == "violin":
            sns.violinplot(
                data=individual_stats,
                x="group",
                y="firing_rate",
                ax=ax,
                palette=color_map,
                inner="box",
                alpha=0.4,
                cut=0,  # This prevents the KDE from extending beyond the data range
            )
        elif pt == "swarm":
            sns.swarmplot(
                data=individual_stats,
                x="group",
                y="firing_rate",
                ax=ax,
                palette=color_map,
                alpha=swarm_alpha,
            )

    # Formatting
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_xlabel("Population Group")
    ax.set_ylabel("Firing Rate (spikes/s)")
    ax.set_title("Firing Rate Distribution for individual cells")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    return ax
