"""
Want to be able to take multiple plot names in and plot them all at the same time, to save time
https://stackoverflow.com/questions/458209/is-there-a-way-to-detach-matplotlib-plots-so-that-the-computation-can-continue
"""
import re
import statistics

import matplotlib
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython import get_ipython
from neuron import h

from ..util import util
from typing import List, Optional, Tuple, Union, Any, Dict

use_description = """

Plot BMTK models easily.

python -m bmtool.plot
"""


def is_notebook() -> bool:
    """Detect if code is running in a Jupyter notebook environment.

    Returns
    -------
    bool
        True if running in a Jupyter notebook, False otherwise.

    Notes
    -----
    This is used to determine whether to call plt.show() explicitly or
    rely on Jupyter's automatic display functionality.
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def total_connection_matrix(
    config: Optional[str] = None,
    title: Optional[str] = None,
    sources: Optional[str] = None,
    targets: Optional[str] = None,
    sids: Optional[str] = None,
    tids: Optional[str] = None,
    no_prepend_pop: bool = False,
    save_file: Optional[str] = None,
    synaptic_info: str = "0",
    include_gap: bool = True,
) -> None:
    """
    Generate a plot displaying total connections or other synaptic statistics.

    Parameters
    ----------
    config : Optional[str], optional
        Path to a BMTK simulation config file. Default is None.
    title : Optional[str], optional
        Title for the plot. If None, a default title will be used based on `synaptic_info`. Default is None.
    sources : Optional[str], optional
        Comma-separated string of network names to use as sources. Default is None.
    targets : Optional[str], optional
        Comma-separated string of network names to use as targets. Default is None.
    sids : Optional[str], optional
        Comma-separated string of source node identifiers to filter. Default is None.
    tids : Optional[str], optional
        Comma-separated string of target node identifiers to filter. Default is None.
    no_prepend_pop : bool, optional
        If True, don't display population name before sid or tid in the plot. Default is False.
    save_file : Optional[str], optional
        Path to save the plot. If None, plot is not saved. Default is None.
    synaptic_info : str, optional
        Type of information to display:
        - '0': Total connections (default).
        - '1': Mean and standard deviation of connections per target cell.
        - '2': All synapse .mod files used (displays file names).
        - '3': All synapse .json files used (displays file names).
        Default is "0".
    include_gap : bool, optional
        If True, include gap junctions and chemical synapses in the analysis.
        If False, only include chemical synapses. Default is True.

    Returns
    -------
    None
        The function generates and displays/saves a plot.

    Raises
    ------
    Exception
        If `config`, `sources`, or `targets` are not provided.
    """
    if not config:
        raise Exception("config not defined")
    if not sources or not targets:
        raise Exception("Sources or targets not defined")
    sources_list: List[str] = sources.split(",")
    targets_list: List[str] = targets.split(",")
    sids_list: List[str] = []
    if sids:
        sids_list = sids.split(",")
    tids_list: List[str] = []
    if tids:
        tids_list = tids.split(",")
        
    text, num, source_labels, target_labels = util.connection_totals(
        config=config, # type: ignore
        nodes=None, # type: ignore
        edges=None, # type: ignore
        sources=sources_list,
        targets=targets_list,
        sids=sids_list,
        tids=tids_list,
        prepend_pop=not no_prepend_pop, # type: ignore
        synaptic_info=synaptic_info, # type: ignore
        include_gap=include_gap, # type: ignore
    )

    plot_title: str = title if title is not None and title != "" else "Total Connections"
    if synaptic_info == "1":
        plot_title = "Mean and Stdev # of Conn on Target"
    elif synaptic_info == "2":
        plot_title = "All Synapse .mod Files Used"
    elif synaptic_info == "3":
        plot_title = "All Synapse .json Files Used"
        
    plot_connection_info(
        text, num, source_labels, target_labels, plot_title, syn_info=synaptic_info, save_file=save_file
    )
    # No explicit return, function plots or saves a figure.


def percent_connection_matrix(
    config: Optional[str] = None,
    nodes: Optional[Any] = None, 
    edges: Optional[Any] = None, 
    title: Optional[str] = None,
    sources: Optional[str] = None,
    targets: Optional[str] = None,
    sids: Optional[str] = None,
    tids: Optional[str] = None,
    no_prepend_pop: bool = False,
    save_file: Optional[str] = None,
    method: str = "total",
    include_gap: bool = True,
) -> None:
    """
    Generates a plot showing the percent connectivity of a network.

    Parameters
    ----------
    config : Optional[str], optional
        Path to a BMTK simulation config file. Default is None.
    nodes : Optional[Any], optional
        Node data. Typically loaded via config by `util.percent_connections` if not provided. Default is None.
    edges : Optional[Any], optional
        Edge data. Typically loaded via config by `util.percent_connections` if not provided. Default is None.
    title : Optional[str], optional
        Title for the plot. If None, "Percent Connectivity" is used. Default is None.
    sources : Optional[str], optional
        Comma-separated string of source network names. Default is None.
    targets : Optional[str], optional
        Comma-separated string of target network names. Default is None.
    sids : Optional[str], optional
        Comma-separated string of source node identifiers to filter. Default is None.
    tids : Optional[str], optional
        Comma-separated string of target node identifiers to filter. Default is None.
    no_prepend_pop : bool, optional
        If True, do not display population name before sid or tid in plot labels. Default is False.
    save_file : Optional[str], optional
        Path to save the plot. If None, plot is not saved. Default is None.
    method : str, optional
        Method to calculate percent connectivity: 'total', 'uni', or 'bi'
        (for total, unidirectional, or bidirectional connections). Default is "total".
    include_gap : bool, optional
        If True, include gap junctions and chemical synapses.
        If False, only include chemical synapses. Default is True.

    Returns
    -------
    None
        The function generates and displays/saves a plot.
        
    Raises
    ------
    Exception
        If `config`, `sources`, or `targets` are not provided.
    """
    if not config:
        raise Exception("config not defined")
    if not sources or not targets:
        raise Exception("Sources or targets not defined")

    sources_list: List[str] = sources.split(",")
    targets_list: List[str] = targets.split(",")
    sids_list: List[str] = []
    if sids:
        sids_list = sids.split(",")
    tids_list: List[str] = []
    if tids:
        tids_list = tids.split(",")
        
    text, num, source_labels, target_labels = util.percent_connections(
        config=config, # type: ignore
        nodes=nodes, 
        edges=edges, 
        sources=sources_list,
        targets=targets_list,
        sids=sids_list,
        tids=tids_list,
        prepend_pop=not no_prepend_pop, # type: ignore
        method=method, # type: ignore
        include_gap=include_gap, # type: ignore
    )
    
    plot_title: str = title if title is not None and title != "" else "Percent Connectivity"

    plot_connection_info(text, num, source_labels, target_labels, plot_title, save_file=save_file)
    # No explicit return


def probability_connection_matrix(
    config: Optional[str] = None,
    nodes: Optional[Any] = None, 
    edges: Optional[Any] = None, 
    title: Optional[str] = None,
    sources: Optional[str] = None,
    targets: Optional[str] = None,
    sids: Optional[str] = None,
    tids: Optional[str] = None,
    no_prepend_pop: bool = False,
    save_file: Optional[str] = None,
    dist_X: bool = True,
    dist_Y: bool = True,
    dist_Z: bool = True,
    bins: int = 8,
    line_plot: bool = False,
    verbose: bool = False,
    include_gap: bool = True,
) -> None:
    """
    Generates a matrix of plots showing connection probability as a function of distance.

    This function calculates connection probabilities based on distances (defaulting to X, Y, Z dimensions)
    between source and target cell populations.
    The probability is often calculated as (actual connections) / (possible pairs of cells).
    The plot matrix shows these probabilities binned by distance.

    Parameters
    ----------
    config : Optional[str], optional
        Path to a BMTK simulation config file. Default is None.
    nodes : Optional[Any], optional
        Node data. Typically loaded via config. Default is None.
    edges : Optional[Any], optional
        Edge data. Typically loaded via config. Default is None.
    title : Optional[str], optional
        Title for the plot matrix. Default is "Distance Probability Matrix".
    sources : Optional[str], optional
        Comma-separated string of source network names. Default is None.
    targets : Optional[str], optional
        Comma-separated string of target network names. Default is None.
    sids : Optional[str], optional
        Comma-separated string of source node identifiers to filter. Default is None.
    tids : Optional[str], optional
        Comma-separated string of target node identifiers to filter. Default is None.
    no_prepend_pop : bool, optional
        If True, do not display population name before sid or tid. Default is False.
    save_file : Optional[str], optional
        Path to save the plot. If None, plot is not saved. Default is None.
    dist_X : bool, optional
        Include X-dimension in distance calculation. Default is True.
    dist_Y : bool, optional
        Include Y-dimension in distance calculation. Default is True.
    dist_Z : bool, optional
        Include Z-dimension in distance calculation. Default is True.
    bins : int, optional
        Number of bins for histogramming distances. Default is 8.
    line_plot : bool, optional
        If True, plot probability as a line plot; otherwise, use a bar plot. Default is False.
    verbose : bool, optional
        If True, print X and Y data for each subplot. Default is False.
    include_gap : bool, optional
        If True, include gap junctions in the analysis. Default is True.

    Returns
    -------
    None
        The function generates and displays/saves a plot matrix.

    Raises
    ------
    Exception
        If `config`, `sources`, or `targets` are not provided.
    
    Notes
    -----
    The calculation `YY = ns[0] / ns[1]` might lead to division by zero if `ns[1]` 
    (number of possible pairs in a distance bin) is zero. This is currently handled 
    by `np.seterr(divide="ignore", invalid="ignore")`, which means results might contain NaN or Inf.
    The sentinel value check `data[0][0] == -1` from `util.connection_probabilities` is used to exit early
    if the underlying utility function indicates no valid data or an error.
    """
    if not config:
        raise Exception("config not defined")
    if not sources or not targets: # Only one check is needed
        raise Exception("Sources or targets not defined")
        
    sources_list: List[str] = sources.split(",")
    targets_list: List[str] = targets.split(",")
    sids_list: List[str] = []
    if sids:
        sids_list = sids.split(",")
    tids_list: List[str] = []
    if tids:
        tids_list = tids.split(",")

    _, data, source_labels, target_labels = util.connection_probabilities( # _ to indicate throwaway
        config=config, # type: ignore
        nodes=nodes,   
        edges=edges,   
        sources=sources_list,
        targets=targets_list,
        sids=sids_list,
        tids=tids_list,
        prepend_pop=not no_prepend_pop, # type: ignore
        dist_X=dist_X, # type: ignore
        dist_Y=dist_Y, # type: ignore
        dist_Z=dist_Z, # type: ignore
        num_bins=bins, # type: ignore
        include_gap=include_gap, # type: ignore
    )
    
    if not hasattr(data, 'shape') or data.size == 0: 
        print("Warning: No data returned from connection_probabilities. Plot will not be generated.")
        return
    
    first_element = data[0,0]
    if isinstance(first_element, dict) and first_element.get('ns') == -1 and first_element.get('bins') == -1:
        print("Warning: Sentinel value structure {'ns': -1, 'bins': -1} received from connection_probabilities, indicating no valid data or specific error. Plot will not be generated.")
        return
    elif isinstance(first_element, int) and first_element == -1:
        print("Warning: Sentinel value (-1) received from connection_probabilities. Plot will not be generated.")
        return

    np.seterr(divide="ignore", invalid="ignore")
    num_src, num_tar = data.shape
    fig, axes = plt.subplots(nrows=num_src, ncols=num_tar, figsize=(12, 12))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)

    for x_idx in range(num_src): # Renamed x to x_idx
        for y_idx in range(num_tar): # Renamed y to y_idx
            ns = data[x_idx][y_idx]["ns"]
            bins_data = data[x_idx][y_idx]["bins"]

            XX = bins_data[:-1]
            YY = ns[0] / ns[1] if ns[1] != 0 else np.nan # Avoid division by zero explicitly

            if line_plot:
                axes[x_idx, y_idx].plot(XX, YY)
            else:
                axes[x_idx, y_idx].bar(XX, YY)

            if x_idx == num_src - 1:
                axes[x_idx, y_idx].set_xlabel(target_labels[y_idx])
            if y_idx == 0:
                axes[x_idx, y_idx].set_ylabel(source_labels[x_idx])

            if verbose:
                print("Source: [" + source_labels[x_idx] + "] | Target: [" + target_labels[y_idx] + "]")
                print("X:")
                print(XX)
                print("Y:")
                print(YY)

    plot_title_str: str = title if title else "Distance Probability Matrix" # Use plot_title_str
    st = fig.suptitle(plot_title_str, fontsize=14)
    fig.text(0.5, 0.04, "Target", ha="center")
    fig.text(0.04, 0.5, "Source", va="center", rotation="vertical")
    notebook_env = is_notebook() 
    if not notebook_env: 
        if save_file:
             fig.savefig(save_file) # Save before show if not in notebook and save_file is provided
        else:
             fig.show()
    elif save_file: # If in notebook and save_file is provided
        fig.savefig(save_file)


def convergence_connection_matrix(
    config: Optional[str] = None,
    title: Optional[str] = None,
    sources: Optional[str] = None,
    targets: Optional[str] = None,
    sids: Optional[str] = None,
    tids: Optional[str] = None,
    no_prepend_pop: bool = False,
    save_file: Optional[str] = None,
    convergence: bool = True, 
    method: str = "mean+std",
    include_gap: bool = True,
    return_dict: Optional[bool] = None,
) -> Optional[Dict[str, Dict[str, Any]]]:
    """
    Generates a connection plot displaying convergence data (number of incoming connections to target cells).

    This function is a wrapper around `divergence_connection_matrix` with the `convergence` flag set to True.

    Parameters
    ----------
    config : Optional[str], optional
        Path to a BMTK simulation config file. Default is None.
    title : Optional[str], optional
        Title for the plot. Default is None (auto-generated based on method and convergence/divergence).
    sources : Optional[str], optional
        Comma-separated string of source network names. Default is None.
    targets : Optional[str], optional
        Comma-separated string of target network names. Default is None.
    sids : Optional[str], optional
        Comma-separated string of source node identifiers to filter. Default is None.
    tids : Optional[str], optional
        Comma-separated string of target node identifiers to filter. Default is None.
    no_prepend_pop : bool, optional
        If True, do not display population name before sid or tid. Default is False.
    save_file : Optional[str], optional
        Path to save the plot. If None, plot is not saved. Default is None.
    convergence : bool, optional
        Flag indicating convergence calculation. Although a parameter, it's passed as True to the underlying function.
        Default is True.
    method : str, optional
        Statistical method for displaying connection counts:
        'mean', 'min', 'max', 'std' (standard deviation), or 'mean+std'. Default is "mean+std".
    include_gap : bool, optional
        If True, include gap junctions in the analysis. Default is True.
    return_dict : Optional[bool], optional
        If True, return the connection data as a dictionary instead of plotting. Default is None.

    Returns
    -------
    Optional[Dict[str, Dict[str, Any]]]
        A dictionary containing the connection data if `return_dict` is True, otherwise None.
        The structure of the dictionary is {source_label: {target_label: value}}.
        
    Raises
    ------
    Exception
        If `config`, `sources`, or `targets` are not provided by the caller.
    """
    if not config:
        raise Exception("config not defined")
    if not sources or not targets:
        raise Exception("Sources or targets not defined")
    return divergence_connection_matrix(
        config,
        title,
        sources,
        targets,
        sids,
        tids,
        no_prepend_pop,
        save_file,
        True, # Explicitly set convergence to True
        method,
        include_gap=include_gap,
        return_dict=return_dict,
    )


def divergence_connection_matrix(
    config: Optional[str] = None,
    title: Optional[str] = None,
    sources: Optional[str] = None,
    targets: Optional[str] = None,
    sids: Optional[str] = None,
    tids: Optional[str] = None,
    no_prepend_pop: bool = False,
    save_file: Optional[str] = None,
    convergence: bool = False,
    method: str = "mean+std",
    include_gap: bool = True,
    return_dict: Optional[bool] = None,
) -> Optional[Dict[str, Dict[str, Any]]]: 
    """
    Generates a connection plot displaying divergence or convergence data.

    Parameters
    ----------
    config : Optional[str], optional
        Path to a BMTK simulation config file. Default is None.
    title : Optional[str], optional
        Title for the plot. Default is None (auto-generated based on method and type).
    sources : Optional[str], optional
        Comma-separated string of source network names. Default is None.
    targets : Optional[str], optional
        Comma-separated string of target network names. Default is None.
    sids : Optional[str], optional
        Comma-separated string of source node IDs to filter. Default is None.
    tids : Optional[str], optional
        Comma-separated string of target node IDs to filter. Default is None.
    no_prepend_pop : bool, optional
        If True, do not display population name before sid or tid. Default is False.
    save_file : Optional[str], optional
        Path to save the plot. If None, plot is not saved. Default is None.
    convergence : bool, optional
        If True, calculate convergence (incoming connections to target). 
        If False, calculate divergence (outgoing connections from source). Default is False.
    method : str, optional
        Statistical method for display: 
        'mean', 'min', 'max', 'std' (standard deviation), 'mean+std'. Default is "mean+std".
    include_gap : bool, optional
        If True, include gap junctions in the analysis. Default is True.
    return_dict : Optional[bool], optional
        If True, return data as a dictionary instead of plotting. Default is None.

    Returns
    -------
    Optional[Dict[str, Dict[str, Any]]]
        A dictionary containing the connection data if `return_dict` is True, otherwise None.
        The structure of the dictionary is {source_label: {target_label: value}}.

    Raises
    ------
    Exception
        If `config`, `sources`, or `targets` are not provided.
    """
    if not config:
        raise Exception("config not defined")
    if not sources or not targets:
        raise Exception("Sources or targets not defined")
        
    sources_list: List[str] = sources.split(",")
    targets_list: List[str] = targets.split(",")
    sids_list: List[str] = []
    if sids:
        sids_list = sids.split(",")
    tids_list: List[str] = []
    if tids:
        tids_list = tids.split(",")

    syn_info, data, source_labels, target_labels = util.connection_divergence( # type: ignore
        config=config,
        nodes=None, # type: ignore
        edges=None, # type: ignore
        sources=sources_list,
        targets=targets_list,
        sids=sids_list,
        tids=tids_list,
        prepend_pop=not no_prepend_pop,
        convergence=convergence,
        method=method,
        include_gap=include_gap,
    )

    plot_title_str: str 
    if title is None or title == "":
        base_title: str
        if method == "min":
            base_title = "Minimum "
        elif method == "max":
            base_title = "Maximum "
        elif method == "std":
            base_title = "Standard Deviation "
        elif method == "mean":
            base_title = "Mean "
        else: # Default to "mean+std"
            base_title = "Mean + Std "

        if convergence:
            plot_title_str = base_title + "Synaptic Convergence"
        else:
            plot_title_str = base_title + "Synaptic Divergence"
    else:
        plot_title_str = title
        
    if return_dict:
        result_data_dict: Optional[Dict[str,Dict[str,Any]]] = plot_connection_info( 
            syn_info,
            data,
            source_labels,
            target_labels,
            plot_title_str,
            save_file=save_file,
            return_dict=return_dict, 
        )
        return result_data_dict 
    else:
        plot_connection_info(
            syn_info, data, source_labels, target_labels, plot_title_str, save_file=save_file, return_dict=False
        )
        return None


def gap_junction_matrix(
    config: Optional[str] = None,
    title: Optional[str] = None,
    sources: Optional[str] = None,
    targets: Optional[str] = None,
    sids: Optional[str] = None,
    tids: Optional[str] = None,
    no_prepend_pop: bool = False,
    save_file: Optional[str] = None,
    method: str = "convergence",
) -> None:
    """
    Generates a connection plot displaying gap junction data.

    Parameters
    ----------
    config : Optional[str], optional
        Path to a BMTK simulation config file. Default is None.
    title : Optional[str], optional
        Title for the plot. Default is None (auto-generated based on method).
    sources : Optional[str], optional
        Comma-separated string of source network names. Default is None.
    targets : Optional[str], optional
        Comma-separated string of target network names. Default is None.
    sids : Optional[str], optional
        Comma-separated string of source node IDs to filter. Default is None.
    tids : Optional[str], optional
        Comma-separated string of target node IDs to filter. Default is None.
    no_prepend_pop : bool, optional
        If True, do not display population name before sid or tid. Default is False.
    save_file : Optional[str], optional
        Path to save the plot. If None, plot is not saved. Default is None.
    method : str, optional
        Method for calculation: 'convergence' or 'percent'. Default is "convergence".

    Returns
    -------
    None
        The function generates and displays/saves a plot.

    Raises
    ------
    Exception
        If `config`, `sources`, or `targets` are not provided, or if `method` is invalid.
    """
    if not config:
        raise Exception("config not defined")
    if not sources or not targets:
        raise Exception("Sources or targets not defined")
    if method not in ["convergence", "percent"]: 
        raise Exception("Method must be 'convergence' or 'percent'")
        
    sources_list: List[str] = sources.split(",")
    targets_list: List[str] = targets.split(",")
    sids_list: List[str] = []
    if sids:
        sids_list = sids.split(",")
    tids_list: List[str] = []
    if tids:
        tids_list = tids.split(",")
        
    syn_info_arr, data_arr, source_labels_list, target_labels_list = util.gap_junction_connections( # type: ignore
        config=config,
        nodes=None, 
        edges=None, 
        sources=sources_list,
        targets=targets_list,
        sids=sids_list,
        tids=tids_list,
        prepend_pop=not no_prepend_pop,
        method=method,
    )

    def filter_rows( 
        current_syn_info: np.ndarray, 
        current_data: np.ndarray, 
        current_source_labels: List[str], 
        current_target_labels: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Filters out rows in a connectivity matrix that contain only NaN or zero values.
        """
        valid_rows = ~np.all(np.isnan(current_data), axis=1) & ~np.all(current_data == 0, axis=1)
        new_syn_info = current_syn_info[valid_rows]
        new_data = current_data[valid_rows]
        new_source_labels_np = np.array(current_source_labels)[valid_rows] 
        return new_syn_info, new_data, new_source_labels_np, current_target_labels

    def filter_rows_and_columns( 
        syn_info_frc: np.ndarray, 
        data_frc: np.ndarray, 
        source_labels_frc: List[str], 
        target_labels_frc: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """
        Filters out both rows and columns in a connectivity matrix that contain only NaN or zero values.
        """
        f_syn_info, f_data, f_source_labels_np, f_target_labels = filter_rows(
            syn_info_frc, data_frc, source_labels_frc, target_labels_frc
        )
        transposed_syn_info_fr = np.transpose(f_syn_info)
        transposed_data_fr = np.transpose(f_data)
        f_syn_info_col, f_data_col, f_source_labels_col_np, _ = filter_rows(
            transposed_syn_info_fr, transposed_data_fr, f_target_labels, list(f_source_labels_np) 
        )
        final_syn_info = np.transpose(f_syn_info_col)
        final_data = np.transpose(f_data_col)
        final_source_labels = list(f_source_labels_np) 
        final_target_labels = list(f_source_labels_col_np)
        return final_syn_info, final_data, final_source_labels, final_target_labels

    syn_info_filtered, data_filtered, source_labels_filtered, target_labels_filtered = filter_rows_and_columns(
        syn_info_arr, data_arr, source_labels_list, target_labels_list 
    )

    plot_title_str: str = title if title is not None and title != "" else "Gap Junction"
    if method == "convergence":
        plot_title_str += " Syn Convergence"
    elif method == "percent":
        plot_title_str += " Percent Connectivity"
        
    plot_connection_info(syn_info_filtered, data_filtered, source_labels_filtered, target_labels_filtered, plot_title_str, save_file=save_file)


def connection_histogram(
    config: Optional[str] = None,
    nodes: Optional[Any] = None,
    edges: Optional[Any] = None,
    sources: Optional[List[str]] = None, 
    targets: Optional[List[str]] = None, 
    sids: Optional[List[str]] = None,    
    tids: Optional[List[str]] = None,    
    no_prepend_pop: bool = True,
    synaptic_info: str = "0",
    source_cell: Optional[str] = None,
    target_cell: Optional[str] = None,
    include_gap: bool = True,
) -> None:
    """
    Generates histogram of the number of connections individual cells in a
    target population receive from a source population.

    This function utilizes `util.relation_matrix` and a nested helper
    `connection_pair_histogram` to generate the plot.

    Parameters
    ----------
    config : Optional[str], optional
        Path to a BMTK simulation config file. Default is None.
    nodes : Optional[Any], optional
        Node data. Default is None (loaded via config by `util.relation_matrix`).
    edges : Optional[Any], optional
        Edge data. Default is None (loaded via config by `util.relation_matrix`).
    sources : Optional[List[str]], optional
        List of source network names. If None, defaults to an empty list and will raise an error.
    targets : Optional[List[str]], optional
        List of target network names. If None, defaults to an empty list and will raise an error.
    sids : Optional[List[str]], optional
        List of source node identifiers (e.g., pop_name) to filter. Default is None (empty list).
    tids : Optional[List[str]], optional
        List of target node identifiers (e.g., pop_name) to filter. Default is None (empty list).
    no_prepend_pop : bool, optional
        If True, do not display population name before sid or tid in plot labels. Default is True.
    synaptic_info : str, optional
        Passed to `util.relation_matrix`, affects data retrieval. Default is "0".
    source_cell : Optional[str], optional
        Specific source cell type/population name to focus the histogram on.
        Used by the internal `connection_pair_histogram`. Default is None.
    target_cell : Optional[str], optional
        Specific target cell type/population name to focus the histogram on.
        Used by the internal `connection_pair_histogram`. Default is None.
    include_gap : bool, optional
        If True, include gap junctions in the analysis. Default is True.

    Returns
    -------
    None
        The function generates and displays a plot.

    Raises
    ------
    Exception
        If `config` is not provided, or if `sources` or `targets` are effectively empty after processing.
    """

    def connection_pair_histogram(**kwargs: Any) -> None: # Added type hint for kwargs
        """
        Creates a histogram showing the distribution of connection counts 
        between a specific source and target cell type.
        (Docstring for nested function improved for clarity)
        """
        edges_df: pd.DataFrame = kwargs["edges"] 
        source_id_type: str = kwargs["sid"]
        target_id_type: str = kwargs["tid"]
        source_id_val: str = kwargs["source_id"] 
        target_id_val: str = kwargs["target_id"] 

        if source_id_val == source_cell and target_id_val == target_cell:
            temp_df = edges_df[
                (edges_df[source_id_type] == source_id_val) & (edges_df[target_id_type] == target_id_val)
            ]
            if not include_gap:
                temp_df = temp_df[~temp_df["is_gap_junction"]]
            
            node_pairs = temp_df.groupby("target_node_id")["source_node_id"].count()
            label_str: str # Renamed label to label_str
            
            if not node_pairs.empty:
                try:
                    conn_mean = statistics.mean(node_pairs.values)
                    conn_std = statistics.stdev(node_pairs.values)
                    conn_median = statistics.median(node_pairs.values)
                    label_str = "mean {:.2f} std {:.2f} median {:.2f}".format(
                        conn_mean, conn_std, conn_median
                    )
                except statistics.StatisticsError: 
                    conn_mean = statistics.mean(node_pairs.values)
                    conn_median = statistics.median(node_pairs.values)
                    label_str = "mean {:.2f} median {:.2f} (N={})".format(conn_mean, conn_median, len(node_pairs.values))
                plt.hist(node_pairs.values, density=False, bins="auto", stacked=True, label=label_str)
            else:
                label_str = "No connections found"
                plt.hist([], bins="auto", label=label_str) 
            
            plt.legend()
            plt.xlabel(f"# of conns from {source_cell} to {target_cell}")
            plt.ylabel("# of cells")
            if not is_notebook(): # Only call plt.show() if not in notebook
                 plt.show()
        else: 
            pass

    if not config:
        raise Exception("config not defined")

    sources_list_cih = sources if sources is not None else []
    targets_list_cih = targets if targets is not None else []
    sids_list_cih = sids if sids is not None else []
    tids_list_cih = tids if tids is not None else []

    if not sources_list_cih : 
        raise Exception("Sources not defined or empty")
    if not targets_list_cih:
        raise Exception("Targets not defined or empty")
        
    util.relation_matrix( # type: ignore
        config=config, 
        nodes=nodes, 
        edges=edges, 
        sources=sources_list_cih, 
        targets=targets_list_cih, 
        sids=sids_list_cih,       
        tids=tids_list_cih,       
        prepend_pop=not no_prepend_pop, 
        relation_func=connection_pair_histogram, 
        synaptic_info=synaptic_info
    )


def connection_distance(
    config: str,
    sources: str,
    targets: str,
    source_cell_id: int,
    target_id_type: str,
    ignore_z: bool = False,
) -> None:
    """
    Plots the 3D spatial distribution of target nodes relative to a source node
    and a histogram of distances from the source node to each target node.

    Parameters
    ----------
    config : str
        Path to a BMTK simulation config file.
    sources : str
        Name of the source network.
    targets : str
        Name of the target network.
    source_cell_id : int
        ID of the source cell for calculating distances to target nodes.
    target_id_type : str
        A string to filter target nodes based on the 'target_query' column in the edges file.
    ignore_z : bool, optional
        If True, ignore the Z-axis when calculating distances (for 2D projections). Default is False.

    Returns
    -------
    None
        The function generates and displays plots.
        
    Raises
    ------
    Exception
        If `config`, `sources`, or `targets` are not provided.
    KeyError
        If specified network names are not found in the loaded node/edge data.
    """
    if not config:
        raise Exception("config not defined")
    if not sources or not targets:
        raise Exception("Sources or targets not defined")

    nodes_all_networks, edges_all_networks = util.load_nodes_edges_from_config(config) # type: ignore

    edge_network_key = f"{sources}_to_{targets}"
    node_network_key = sources 

    if node_network_key not in nodes_all_networks:
        raise KeyError(f"Node network '{node_network_key}' not found in loaded nodes.")
    if edge_network_key not in edges_all_networks:
        raise KeyError(f"Edge network '{edge_network_key}' not found in loaded edges.")

    current_nodes_df: pd.DataFrame = nodes_all_networks[node_network_key] # type: ignore
    current_edges_df: pd.DataFrame = edges_all_networks[edge_network_key] # type: ignore

    source_edges_df = current_edges_df[current_edges_df["source_node_id"] == source_cell_id]
    if target_id_type: # Ensure target_id_type is not empty before trying to filter
        source_edges_df = source_edges_df[source_edges_df["target_query"].str.contains(target_id_type, na=False)]

    target_node_ids_arr = source_edges_df["target_node_id"].unique() # Use unique to avoid redundant calculations

    if target_node_ids_arr.size == 0:
        print(f"No target nodes found for source_cell_id {source_cell_id} and target_id_type '{target_id_type}'.")
        return

    target_nodes_df = current_nodes_df.loc[current_nodes_df.index.isin(target_node_ids_arr)]
    source_node_df = current_nodes_df.loc[current_nodes_df.index == source_cell_id]

    if source_node_df.empty:
        print(f"Warning: Source cell ID {source_cell_id} not found in network {node_network_key}.")
        return
    if target_nodes_df.empty:
        print(f"Warning: No target cells found for source {source_cell_id} with target type {target_id_type} after filtering nodes.")
        return
        
    pos_columns = ["pos_x", "pos_y", "pos_z"] if not ignore_z else ["pos_x", "pos_y"]
    
    target_positions = target_nodes_df[pos_columns].values
    source_position = source_node_df[pos_columns].values.ravel()

    if target_positions.shape[0] == 0:
        print("No target positions to calculate distance from.")
        distances = np.array([])
    else:
        distances = np.linalg.norm(target_positions - source_position, axis=1)

    fig = plt.figure(figsize=(8, 6))
    if ignore_z:
        ax = fig.add_subplot(111)
        ax.scatter(target_nodes_df["pos_x"], target_nodes_df["pos_y"], c="blue", label="Target Cells")
        ax.scatter(source_node_df["pos_x"].iloc[0], source_node_df["pos_y"].iloc[0], c="red", label="Source Cell", marker='x', s=100)
    else:
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            target_nodes_df["pos_x"], target_nodes_df["pos_y"], target_nodes_df["pos_z"],
            c="blue", label="Target Cells"
        )
        ax.scatter(
            source_node_df["pos_x"].iloc[0], source_node_df["pos_y"].iloc[0], source_node_df["pos_z"].iloc[0],
            c="red", label="Source Cell", marker='x', s=100 # type: ignore
        )
        ax.set_zlabel("Z position (um)") # type: ignore
    
    ax.set_xlabel("X position (um)")
    ax.set_ylabel("Y position (um)")
    ax.legend()
    ax.set_title(f"Spatial Distribution (Source ID: {source_cell_id})")
    
    if not is_notebook():
        plt.show()

    if distances.size > 0:
        plt.figure(figsize=(8, 6))
        plt.hist(distances, bins=20, color="blue", edgecolor="black")
        plt.xlabel("Distance (um)") 
        plt.ylabel("Number of Target Cells")
        plt.title(f"Distribution of Distances from Source ID: {source_cell_id}")
        plt.grid(True, alpha=0.5)
        if not is_notebook():
            plt.show()
    else:
        print("No distances to plot for the histogram.")


def edge_histogram_matrix(
    config: Optional[str] = None,
    sources: Optional[str] = None,
    targets: Optional[str] = None,
    sids: Optional[str] = None,
    tids: Optional[str] = None,
    no_prepend_pop: bool = False, 
    edge_property: Optional[str] = None,
    time: Optional[int] = None,
    time_compare: Optional[int] = None,
    report: Optional[str] = None,
    title: Optional[str] = None,
    save_file: Optional[str] = None,
) -> None:
    """
    Generates a matrix of histograms showing the distribution of edge properties
    between different populations.

    Each cell in the grid represents the distribution of a specific edge property
    (e.g., 'syn_weight', 'delay') between a source population (row) and
    a target population (column).

    Parameters
    ----------
    config : Optional[str], optional
        Path to a BMTK simulation config file. Default is None.
    sources : Optional[str], optional
        Comma-separated list of source network names. Default is None.
    targets : Optional[str], optional
        Comma-separated list of target network names. Default is None.
    sids : Optional[str], optional
        Comma-separated list of source node identifiers to filter by. Default is None.
    tids : Optional[str], optional
        Comma-separated list of target node identifiers to filter by. Default is None.
    no_prepend_pop : bool, optional
        If True, population names are not prepended to node identifiers in display. Default is False.
    edge_property : Optional[str], optional
        The edge property to analyze (e.g., 'syn_weight', 'delay'). Default is None.
    time : Optional[int], optional
        Time point (e.g., frame or step number) to analyze from a time-series report if applicable. Default is None.
    time_compare : Optional[int], optional
        Second time point for comparison with `time`, if applicable. Default is None.
    report : Optional[str], optional
        Name of the specific report to analyze if edges are stored in sub-reports. Default is None.
    title : Optional[str], optional
        Custom title for the plot. Default is auto-generated based on `edge_property`.
    save_file : Optional[str], optional
        Path to save the generated plot. If None, plot is displayed. Default is None.

    Returns
    -------
    None
        Displays or saves a matrix of histograms.
        
    Raises
    ------
    Exception
        If `config`, `sources`, or `targets` are not provided.
        If `edge_property` is not provided.
    """

    if not config:
        raise Exception("config not defined")
    if not sources or not targets:
        raise Exception("Sources or targets not defined")
    if not edge_property: 
        raise Exception("edge_property must be specified")

    sources_list: List[str] = sources.split(",")
    targets_list: List[str] = targets.split(",")
    
    sids_list: List[str] = []
    if sids:
        sids_list = sids.split(",")
    tids_list: List[str] = []
    if tids:
        tids_list = tids.split(",")
    
    data, source_labels, target_labels = util.edge_property_matrix( # type: ignore
        property_name=edge_property, 
        nodes=None, 
        edges=None, 
        config=config,
        sources=sources_list,
        targets=targets_list,
        sids=sids_list,
        tids=tids_list,
        prepend_pop=not no_prepend_pop, 
        report_name=report, 
        t_window=time, 
        t_compare_window=time_compare, 
    )

    num_src, num_tar = data.shape
    fig, axes = plt.subplots(nrows=num_src, ncols=num_tar, figsize=(12, 12))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)

    for x_idx in range(num_src): 
        for y_idx in range(num_tar): 
            axes[x_idx, y_idx].hist(data[x_idx][y_idx])

            if x_idx == num_src - 1:
                axes[x_idx, y_idx].set_xlabel(target_labels[y_idx])
            if y_idx == 0:
                axes[x_idx, y_idx].set_ylabel(source_labels[x_idx])

    plot_title_str: str 
    if title:
        plot_title_str = title
    else:
        plot_title_str = f"{edge_property} Histogram Matrix"
        if report:
            plot_title_str += f" (Report: {report})"
        if time is not None:
            plot_title_str += f" (Time: {time})"
            if time_compare is not None:
                plot_title_str += f" vs {time_compare}"

    st = fig.suptitle(plot_title_str, fontsize=14)
    fig.text(0.5, 0.04, "Target", ha="center")
    fig.text(0.04, 0.5, "Source", va="center", rotation="vertical")
    plt.draw()
    if save_file:
        plt.savefig(save_file)
    if not is_notebook():
        plt.show()


def plot_synapse_location_histograms(
    config: str, 
    target_model: str, 
    source: Optional[str] = None, 
    target: Optional[str] = None
) -> None:
    """
    Generates a histogram of the positions of synapses on a target cell's sections,
    colored by the source population of the presynaptic cell.

    Parameters
    ----------
    config : str
        Path to a BMTK simulation configuration file.
    target_model : str
        The name of the `model_template` (e.g., "ctdb:Biophys1.hoc") 
        used for the target cell type when building the BMTK node.
    source : Optional[str], optional
        The source BMTK network name (e.g., "v1"). If None, an error will be raised
        if the network key cannot be inferred or is missing.
    target : Optional[str], optional
        The target BMTK network name (e.g., "v1"). If None, an error will be raised
        if the network key cannot be inferred or is missing.
    
    Returns
    -------
    None
        Displays a plot and prints a summary table.

    Notes
    -----
    This function requires NEURON (h object from `neuron` import) to be available and 
    that `util.load_templates_from_config(config)` can successfully load
    the specified `target_model`. It also assumes `nodes` and `edges` are structured
    as dictionaries keyed by network names by `util.load_nodes_edges_from_config`.
    """
    if not source or not target:
        raise ValueError("Both source and target network names must be provided.")

    util.load_templates_from_config(config)

    nodes_all_networks, edges_all_networks = util.load_nodes_edges_from_config(config) # type: ignore
    
    current_nodes: pd.DataFrame = nodes_all_networks[source] # type: ignore
    edge_key = f"{source}_to_{target}"
    if edge_key not in edges_all_networks:
        raise KeyError(f"Edges from '{source}' to '{target}' not found in edge data.")
    current_edges: pd.DataFrame = edges_all_networks[edge_key] # type: ignore

    current_edges["target_model_template"] = current_edges["target_node_id"].map(current_nodes["model_template"])
    current_edges["source_pop_name"] = current_edges["source_node_id"].map(current_nodes["pop_name"])
    current_edges = current_edges[current_edges["target_model_template"] == target_model]

    if current_edges.empty:
        print(f"No edges found for target model {target_model} from source {source} to target {target}.")
        return

    model_name_for_neuron = target_model.split(":")[1] if ":" in target_model else target_model
    try:
        cell = getattr(h, model_name_for_neuron)()
    except AttributeError:
        raise AttributeError(f"NEURON object h does not have attribute {model_name_for_neuron}. Ensure NEURON mechanisms and model templates are loaded.")

    section_id_to_name: Dict[int, str] = {idx: sec.name() for idx, sec in enumerate(cell.all)} # type: ignore
    current_edges["afferent_section_name"] = current_edges["afferent_section_id"].map(section_id_to_name)

    unique_pops = current_edges["source_pop_name"].unique()
    section_counts = current_edges["afferent_section_name"].value_counts()
    sections_with_data = section_counts[section_counts > 0].index.tolist()

    if not sections_with_data:
        print(f"No synaptic data found on sections for target model {target_model}.")
        return

    fig, axes = plt.subplots(nrows=len(sections_with_data), ncols=1, figsize=(8, 2.5 * len(sections_with_data)), squeeze=False)
    axes = axes.ravel() 

    cmap = plt.get_cmap('tab10', len(unique_pops) if len(unique_pops) > 0 else 1)
    pop_colors: Dict[str, Any] = {pop: cmap(i) for i, pop in enumerate(unique_pops)}

    for i, section_name_str in enumerate(sections_with_data):
        ax = axes[i]
        section_data_df = current_edges[current_edges["afferent_section_name"] == section_name_str]
        
        for pop_name, pop_group_df in section_data_df.groupby("source_pop_name"):
            if len(pop_group_df) > 0:
                ax.hist(
                    pop_group_df["afferent_section_pos"],
                    bins=15, alpha=0.7, label=str(pop_name), color=pop_colors.get(str(pop_name))
                )
        ax.set_title(f"{section_name_str}", fontsize=10)
        ax.set_xlabel("Synapse Position on Section (0.0 to 1.0)", fontsize=8)
        ax.set_ylabel("Number of Synapses", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8, title="Source Population")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle(f"Synapse Distribution on {target_model} by Section and Source Population", fontsize=16)
    
    if not is_notebook():
        plt.show()
    # In a notebook, plt.show() is often not needed if %matplotlib inline is used.
    # However, explicit plt.show() can be useful in some contexts or if the magic isn't used.
    # Forcing it here for consistency, but it might be redundant in notebooks.
    elif is_notebook():
        plt.show()


    print("\nSummary of connections by section and source population:")
    try:
        pivot_table = current_edges.pivot_table(
            values="afferent_section_pos", # Value column for aggfunc='count'
            index="afferent_section_name",
            columns="source_pop_name",
            aggfunc="count",
            fill_value=0,
        )
        print(pivot_table)
    except Exception as e:
        print(f"Could not generate pivot table: {e}")


def plot_connection_info(
    text_matrix: np.ndarray, 
    num_matrix: np.ndarray, 
    source_labels: List[str], 
    target_labels: List[str], 
    title_str: str, # Renamed title to title_str
    syn_info_str: str = "0", # Renamed syn_info
    save_file_path: Optional[str] = None, # Renamed save_file
    return_plot_dict: Optional[bool] = None # Renamed return_dict
) -> Optional[Dict[str, Dict[str, Any]]]:
    """
    Plots connection information as a heatmap.

    Parameters
    ----------
    text_matrix : np.ndarray
        Matrix containing the text to display in each cell of the heatmap.
    num_matrix : np.ndarray
        Matrix containing numerical values for coloring the heatmap cells.
        NaNs will be converted to 0.
    source_labels : List[str]
        Labels for the rows (source populations).
    target_labels : List[str]
        Labels for the columns (target populations).
    title_str : str
        Title for the plot.
    syn_info_str : str, optional
        Type of synaptic information, affects text display.
        '2' or '3' rotate text for better readability of file names. Default is "0".
    save_file_path : Optional[str], optional
        If provided, path to save the plot. Default is None.
    return_plot_dict : Optional[bool], optional
        If True, return the connection data as a dictionary. Default is None.

    Returns
    -------
    Optional[Dict[str, Dict[str, Any]]]
        A dictionary of the plotted data if `return_plot_dict` is True, else None.
        Structure: {source_label: {target_label: text_value}}.
    """
    num_source = len(source_labels)
    num_target = len(target_labels)

    matplotlib.rc("image", cmap="viridis")

    base_width_per_col = 2.0
    base_height_per_row = 1.5
    min_width, max_width = 6.0, 16.0
    min_height, max_height = 4.0, 12.0

    desired_width = max(min_width, min(max_width, num_target * base_width_per_col))
    desired_height = max(min_height, min(max_height, num_source * base_height_per_row))

    if num_source <= 2 and num_target <= 4:
        desired_width = max(8.0, desired_width)
        desired_height = max(6.0, desired_height)
    elif num_source <= 3 and num_target <= 3:
        desired_width = max(7.0, desired_width)
        desired_height = max(5.0, desired_height)

    fig, ax = plt.subplots(figsize=(desired_width, desired_height)) # Renamed fig1, ax1
    processed_num_matrix = np.nan_to_num(num_matrix, nan=0.0) # Ensure float for nan=0.0
    ax.imshow(processed_num_matrix)

    ax.set_xticks(np.arange(len(target_labels)))
    ax.set_yticks(np.arange(len(source_labels)))
    ax.set_xticklabels(target_labels, weight="semibold") # Removed size for auto-adjustment
    ax.set_yticklabels(source_labels, weight="semibold") # Removed size

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    graph_data_dict: Dict[str, Dict[str, Any]] = {} # Renamed graph_dict

    text_size: float
    if num_source * num_target <= 4: text_size = 14
    elif num_source * num_target <= 16: text_size = 12
    elif num_source > 20 or num_target > 20: text_size = 7
    elif num_source > 8 or num_target > 8: text_size = 8
    else: text_size = 11

    for i in range(num_source):
        for j in range(num_target):
            edge_info = text_matrix[i, j] if text_matrix[i, j] is not None else "0" # Default to "0" string
            if source_labels[i] not in graph_data_dict:
                graph_data_dict[source_labels[i]] = {}
            graph_data_dict[source_labels[i]][target_labels[j]] = edge_info

            current_text_size = text_size
            rotation = 0
            if syn_info_str == "2" or syn_info_str == "3":
                rotation = 37.5
                if num_source > 8 and num_source < 20: current_text_size = max(text_size - 2, 6)
                elif num_source > 20: current_text_size = max(text_size - 3, 5)
            
            ax.text(
                j, i, edge_info, ha="center", va="center", color="w", 
                size=current_text_size, weight="semibold", rotation=rotation
            )

    ax.set_ylabel("Source", size=13, weight="semibold")
    ax.set_xlabel("Target", size=13, weight="semibold")
    ax.set_title(title_str, size=16, weight="semibold")
    plt.tight_layout(pad=2.0)

    if save_file_path:
        plt.savefig(save_file_path, dpi=300, bbox_inches="tight")

    if not is_notebook() and not save_file_path: # Show only if not saving and not in notebook
        plt.show()
    elif is_notebook() and not save_file_path: # Ensure plot shows in notebook if not saved
        plt.show()


    if return_plot_dict:
        return graph_data_dict
    return None


def connector_percent_matrix(
    csv_path: Optional[str] = None, # Made Optional
    exclude_strings: Optional[List[str]] = None,
    assemb_key: Optional[str] = None,
    title: str = "Percent connection matrix",
    pop_order: Optional[List[str]] = None,
) -> None:
    """
    Generates and plots a connection matrix based on connection probabilities from a CSV file
    produced by bmtool.connector.

    Parameters
    ----------
    csv_path : Optional[str], optional
        Path to the CSV file containing the connection data. Expected output from
        `bmtool.connector.ConnCalculator.save_connection_report()`. Default is None.
    exclude_strings : Optional[List[str]], optional
        List of strings. Rows where 'Source' or 'Target' contain any of these strings will be excluded.
        Default is None.
    assemb_key : Optional[str], optional
        A key string to identify rows related to "assemblies". If found in 'Source',
        these rows are processed to calculate mean probabilities and combined. Default is None.
    title : str, optional
        Title for the generated plot. Default is "Percent connection matrix".
    pop_order : Optional[List[str]], optional
        List of population labels to specify the order for plot axes. Default is None (alphanumerical sorting).

    Returns
    -------
    None
        Displays a heatmap plot of the connection matrix.
        
    Raises
    ------
    FileNotFoundError
        If `csv_path` is not found.
    Exception
        If the CSV format is unsupported or data processing fails.
    """
    if csv_path is None:
        raise ValueError("csv_path must be provided.")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found at path: {csv_path}")

    selected_column = "Percent connectionivity within possible connections" # Typo "connectivity"
    if selected_column not in df.columns:
         selected_column = "Percent connectivity within possible connections" # Try correct spelling
         if selected_column not in df.columns:
            raise KeyError(f"Required column '{selected_column}' (or with typo) not found in CSV.")


    def filter_dataframe(df_local: pd.DataFrame, column_name: str, exclude_list: Optional[List[str]], assembly_key_local: Optional[str]) -> pd.DataFrame:
        def process_string(s: Any) -> Optional[str]: # Input can be Any from CSV
            if not isinstance(s, str): # Handle non-string entries if any
                return None
            
            match = re.search(r"\[\'(.*?)\'\]", s) # Original regex
            
            if exclude_list and any(ex_str in s for ex_str in exclude_list):
                return None
            
            processed_s = s
            if match:
                processed_s = match.group(1)
            
            if "Gap" in s: # Check original string for "Gap"
                processed_s = processed_s + "-Gap"

            if assembly_key_local and assembly_key_local in s: # Check original string for assembly key
                processed_s = processed_s + assembly_key_local # Append key to distinguish
            
            return processed_s

        df_local[column_name] = df_local[column_name].apply(process_string)
        return df_local.dropna(subset=[column_name])

    df = filter_dataframe(df, "Source", exclude_strings, assemb_key)
    df = filter_dataframe(df, "Target", exclude_strings, assemb_key)

    if assemb_key:
        # Logic for processing assemblies - ensure it handles cases where assemb_key might not be present in all targets
        # This part is complex and highly specific to the CSV structure, might need more context for full validation
        processed_assembly_rows = []
        assembly_sources = df[df["Source"].str.contains(assemb_key, na=False)]["Source"].unique()

        for source_name in assembly_sources:
            source_assemblies_df = df[df["Source"] == source_name]
            unique_targets_for_source = source_assemblies_df["Target"].unique()

            for target_name in unique_targets_for_source:
                current_assembly_group = source_assemblies_df[source_assemblies_df["Target"] == target_name]
                if current_assembly_group.empty:
                    continue

                forward_probs = []
                for _, row_data in current_assembly_group.iterrows():
                    perc_val = row_data[selected_column]
                    try:
                        if isinstance(perc_val, str):
                            # Expecting space-separated floats within brackets, e.g., "[0.1 0.2]"
                            str_values = perc_val.strip("[]").split()
                            num_values = [float(p) for p in str_values if p] # Filter out empty strings from multiple spaces
                        elif isinstance(perc_val, (int, float)):
                            num_values = [float(perc_val)]
                        else:
                            continue # Skip if not parsable
                            
                        if len(num_values) >= 1: forward_probs.append(num_values[0])
                        if len(num_values) == 3: forward_probs.append(num_values[1]) # Special handling for 3 values
                    except ValueError:
                        print(f"Warning: Could not parse percentage value '{perc_val}' for {source_name}->{target_name}")
                        continue
                
                if forward_probs:
                    mean_prob = np.mean(forward_probs)
                    # Create new row with processed names (without assemb_key for grouping)
                    new_source_name = source_name.replace(assemb_key, "")
                    new_target_name = target_name.replace(assemb_key, "") if isinstance(target_name, str) else target_name
                    
                    processed_assembly_rows.append({
                        "Source": new_source_name,
                        "Target": new_target_name,
                        selected_column: mean_prob,
                        # Assuming other columns might be needed or can be NaN/0
                        "Percent connectivity within all connections": 0 
                    })
        
        if processed_assembly_rows:
            # Remove original assembly rows and add processed ones
            df = df[~df["Source"].str.contains(assemb_key, na=False)]
            df = pd.concat([df, pd.DataFrame(processed_assembly_rows)], ignore_index=True)


    connection_data: Dict[Tuple[str, str], List[float]] = {}
    for _, row in df.iterrows():
        source_val, target_val, perc_val = row["Source"], row["Target"], row[selected_column]
        
        val_to_store: List[float]
        try:
            if isinstance(perc_val, str):
                str_values = perc_val.strip("[]").split()
                val_to_store = [float(p) for p in str_values if p]
            elif isinstance(perc_val, (int, float)):
                val_to_store = [float(perc_val)]
            else: # Skip if not parsable
                print(f"Warning: Skipping unparsable percentage value '{perc_val}' for {source_val}->{target_val}")
                continue
        except ValueError:
            print(f"Warning: Could not parse percentage value '{perc_val}' for {source_val}->{target_val}")
            continue

        connection_data[(str(source_val), str(target_val))] = val_to_store


    current_populations: List[str] = sorted(list(set(df["Source"].unique()) | set(df["Target"].unique())))
    if pop_order:
        current_populations = [pop for pop in pop_order if pop in current_populations]
    
    num_populations = len(current_populations)
    if num_populations == 0:
        print("No populations to plot after filtering.")
        return

    connection_matrix = np.zeros((num_populations, num_populations), dtype=float)
    for (s, t), probabilities_list in connection_data.items():
        if s in current_populations and t in current_populations:
            s_idx, t_idx = current_populations.index(s), current_populations.index(t)
            # Handle different formats of probabilities_list based on original logic
            if probabilities_list: # Ensure not empty
                if len(probabilities_list) == 1 or len(probabilities_list) == 2: # Takes first for uni/total if 2 are present (e.g. fwd/bwd)
                    connection_matrix[s_idx, t_idx] = probabilities_list[0]
                elif len(probabilities_list) == 3: # Special case for bi-directional from connector?
                    connection_matrix[s_idx, t_idx] = probabilities_list[0]
                    if t_idx < num_populations and s_idx < num_populations : # Check bounds for reverse if applicable
                         connection_matrix[t_idx, s_idx] = probabilities_list[1] 
                elif isinstance(probabilities_list, float): # Should not happen due to list conversion but as fallback
                     connection_matrix[s_idx, t_idx] = probabilities_list
            # else: print warning or handle? For now, it defaults to 0.

    fig, ax = plt.subplots(figsize=(max(8, num_populations * 0.8), max(6, num_populations * 0.6)))
    im = ax.imshow(connection_matrix, cmap="viridis", interpolation="nearest", vmin=0, vmax=100) # Assuming percentages

    for i in range(num_populations):
        for j in range(num_populations):
            ax.text(
                j, i, f"{connection_matrix[i, j]:.1f}%", # One decimal place for percentages
                ha="center", va="center", color="w", size=max(6, 10 - num_populations // 3), weight="semibold"
            )

    plt.colorbar(im, label=selected_column.replace("connectionivity", "connectivity"))
    ax.set_title(title)
    ax.set_xlabel("Target Population")
    ax.set_ylabel("Source Population")
    ax.set_xticks(np.arange(num_populations))
    ax.set_yticks(np.arange(num_populations))
    ax.set_xticklabels(current_populations, rotation=45, ha="right", size=10, weight="semibold")
    ax.set_yticklabels(current_populations, size=10, weight="semibold")
    plt.tight_layout()
    
    if not is_notebook():
        plt.show()
    elif is_notebook(): # Ensure plot shows in notebook
        plt.show()


def plot_3d_positions(
    config: Optional[str] = None, 
    sources: Optional[str] = None, 
    sid: Optional[str] = None, 
    title: Optional[str] = None, 
    save_file: Optional[str] = None, 
    subset: Optional[int] = None
) -> Optional[plt.Axes]:
    """
    Plots a 3D graph of cell positions based on network and grouping.

    Parameters
    ----------
    config : Optional[str], optional
        Path to a BMTK simulation config file. Default is None.
    sources : Optional[str], optional
        Comma-separated string of network names to plot. If "all" or None, all networks are plotted. Default is None.
    sid : Optional[str], optional
        Column name in the nodes file to use for grouping cells by color/label (e.g., 'pop_name', 'node_type_id'). 
        If multiple networks are plotted, this should be a comma-separated list of column names, one for each network.
        Defaults to 'node_type_id' for each network if not specified or if list is shorter than networks list.
    title : Optional[str], optional
        Title for the plot. Default is "3D positions".
    save_file : Optional[str], optional
        Path to save the plot. If None, plot is displayed. Default is None.
    subset : Optional[int], optional
        If provided, plot every Nth cell to reduce density for large networks. Default is None (plot all).

    Returns
    -------
    Optional[plt.Axes]
        The Matplotlib 3D Axes object if data was plotted, otherwise None.

    Raises
    ------
    Exception
        If `config` is not provided.
        If a specified `sid` column is not found in the corresponding node data.
        If more `sid`s are provided than networks when `sources` is not "all".
    """
    if not config:
        raise Exception("config not defined")

    sources_list: List[str]
    if sources is None or sources.lower() == "all":
        nodes_all_networks = util.load_nodes_from_config(config) # type: ignore
        sources_list = list(nodes_all_networks.keys())
    else:
        sources_list = sources.split(",")
        nodes_all_networks = util.load_nodes_from_config(config, networks=sources_list) # type: ignore
        
    group_keys_list: List[str] = []
    if sid:
        group_keys_list = sid.split(",")
    
    # Ensure group_keys_list matches length of sources_list, padding with default if necessary
    if len(group_keys_list) < len(sources_list):
        group_keys_list.extend(["node_type_id"] * (len(sources_list) - len(group_keys_list)))
    elif len(group_keys_list) > len(sources_list) and sources_list: # Avoid error if sources_list is empty (e.g. "all" with no networks)
        # This case might be an error or require specific handling depending on desired behavior
        print(f"Warning: More group_keys ({len(group_keys_list)}) provided than source networks ({len(sources_list)}). Extra keys will be ignored or may cause issues if source_list is empty.")
        # For now, will proceed, and zip will truncate, or error if nodes_all_networks is empty.

    plot_title: str = title if title is not None else "3D Cell Positions"

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    handles = []
    
    processed_networks = 0
    for i, net_name in enumerate(sources_list):
        if net_name not in nodes_all_networks:
            print(f"Warning: Network '{net_name}' not found in config. Skipping.")
            continue
        
        nodes_df: pd.DataFrame = nodes_all_networks[net_name] # type: ignore
        group_key_for_net: Optional[str] = group_keys_list[i] if i < len(group_keys_list) else "node_type_id" # Fallback

        if group_key_for_net and group_key_for_net not in nodes_df.columns:
            print(f"Warning: Group key '{group_key_for_net}' not found in network '{net_name}'. Plotting all as one group.")
            group_key_for_net = None # Plot all as one group

        unique_groups = nodes_df[group_key_for_net].unique() if group_key_for_net else [None]
        cmap = plt.get_cmap('hsv', len(unique_groups) if len(unique_groups) > 0 else 1) # Ensure cmap has enough colors

        for j, group_name in enumerate(unique_groups):
            group_df = nodes_df[nodes_df[group_key_for_net] == group_name] if group_key_for_net else nodes_df
            
            if not all(col in group_df.columns for col in ["pos_x", "pos_y", "pos_z"]):
                print(f"Warning: Missing position columns in group '{group_name}' for network '{net_name}'. Skipping.")
                continue

            plot_df = group_df.iloc[::subset] if subset is not None and subset > 0 else group_df
            if plot_df.empty:
                continue

            label = f"{net_name}: {group_name}" if group_key_for_net and group_name is not None else net_name
            color_val = cmap(j) if group_key_for_net else "blue" # Fallback color if no grouping

            h = ax.scatter(
                plot_df["pos_x"], plot_df["pos_y"], plot_df["pos_z"],
                color=color_val, label=label, alpha=0.7, s=10 # Added alpha and size
            )
            handles.append(h)
        processed_networks +=1

    if not handles:
        print("No data to plot.")
        plt.close(fig) # Close empty figure
        return None

    ax.set_xlabel("X Position (um)")
    ax.set_ylabel("Y Position (um)")
    ax.set_zlabel("Z Position (um)") # type: ignore
    ax.set_title(plot_title)
    
    # Create a legend with unique labels
    unique_labels_handles = {}
    for h_item in handles:
        if h_item.get_label() not in unique_labels_handles:
             unique_labels_handles[h_item.get_label()] = h_item # type: ignore
    if unique_labels_handles:
        ax.legend(handles=unique_labels_handles.values(), loc='center left', bbox_to_anchor=(1.05, 0.5), borderaxespad=0.)


    plt.draw()
    if save_file:
        plt.savefig(save_file, dpi=300, bbox_inches="tight")
    
    if not is_notebook():
        plt.show()
    elif is_notebook() and not save_file: # Show in notebook only if not saving
         plt.show()

    return ax


def plot_3d_cell_rotation(
    config: Optional[str] = None,
    sources: Optional[str] = None, # Comma-separated string or "all"
    sids: Optional[str] = None,    # Comma-separated string of group_by keys
    title: Optional[str] = None,
    save_file: Optional[str] = None,
    quiver_length: Optional[float] = None, # Added type hint
    arrow_length_ratio: Optional[float] = None, # Added type hint
    group: Optional[str] = None, # Comma-separated string of group names to include
    subset: Optional[int] = None, # Plot every Nth cell
) -> Optional[plt.Axes]:
    from scipy.spatial.transform import Rotation as R

    if not config:
        raise Exception("config not defined")

    sources_list: List[str]
    nodes_all_networks = util.load_nodes_from_config(config) # type: ignore
    if sources is None or sources.lower() == "all":
        sources_list = list(nodes_all_networks.keys())
    else:
        sources_list = sources.split(",")
        # Filter nodes_all_networks to only load specified networks if necessary, though load_nodes_from_config might do this
        # For simplicity, assuming util.load_nodes_from_config handles filtering or loads all and we select here.

    group_keys_list: List[str] = sids.split(",") if sids else []
    
    if len(group_keys_list) < len(sources_list):
        group_keys_list.extend(["node_type_id"] * (len(sources_list) - len(group_keys_list)))

    plot_title: str = title if title is not None else "Cell Rotations"
    
    # Determine default quiver length based on overall spatial extent if not provided
    all_x, all_y, all_z = [], [], []
    for net_name in sources_list:
        if net_name in nodes_all_networks:
            net_nodes: pd.DataFrame = nodes_all_networks[net_name] # type: ignore
            if "pos_x" in net_nodes: all_x.extend(net_nodes["pos_x"].dropna())
            if "pos_y" in net_nodes: all_y.extend(net_nodes["pos_y"].dropna())
            if "pos_z" in net_nodes: all_z.extend(net_nodes["pos_z"].dropna())
    
    if quiver_length is None and all_x and all_y and all_z: # Auto-calculate quiver_length
        max_extent = max(np.ptp(all_x), np.ptp(all_y), np.ptp(all_z)) if all_x and all_y and all_z else 10.0
        quiver_length = max_extent / 20.0 # Default to 5% of max extent
    elif quiver_length is None:
        quiver_length = 10.0 # Default if no spatial data

    if arrow_length_ratio is None:
        arrow_length_ratio = 0.3


    fig = plt.figure(figsize=(12, 12)) # Increased figure size
    ax = fig.add_subplot(111, projection="3d")
    handles = []
    
    groups_to_plot = group.split(',') if group else None

    for i, net_name in enumerate(sources_list):
        if net_name not in nodes_all_networks:
            print(f"Warning: Network '{net_name}' not found. Skipping.")
            continue
        
        nodes_df: pd.DataFrame = nodes_all_networks[net_name] # type: ignore
        group_key_for_net: Optional[str] = group_keys_list[i] if i < len(group_keys_list) else None

        if group_key_for_net and group_key_for_net not in nodes_df.columns:
            print(f"Warning: Group key '{group_key_for_net}' not found in network '{net_name}'. Using no grouping for this network.")
            group_key_for_net = None
            
        unique_groups_in_df = nodes_df[group_key_for_net].unique() if group_key_for_net else [None]
        cmap = plt.get_cmap('hsv', len(unique_groups_in_df) if len(unique_groups_in_df) > 0 else 1)

        for j, group_name_val in enumerate(unique_groups_in_df):
            if groups_to_plot and group_name_val not in groups_to_plot:
                continue

            group_df = nodes_df[nodes_df[group_key_for_net] == group_name_val] if group_key_for_net else nodes_df
            
            plot_df = group_df.iloc[::subset] if subset is not None and subset > 0 else group_df
            if plot_df.empty:
                continue

            required_cols = ["pos_x", "pos_y", "pos_z", 
                             "rotation_angle_xaxis", "rotation_angle_yaxis", "rotation_angle_zaxis"]
            if not all(col in plot_df.columns for col in required_cols):
                print(f"Warning: Missing required columns for rotation plotting in group '{group_name_val}', network '{net_name}'. Skipping.")
                continue

            X, Y, Z = plot_df["pos_x"], plot_df["pos_y"], plot_df["pos_z"]
            # Euler angles: U (phi, around x), V (theta, around y), W (psi, around z)
            U_phi = plot_df["rotation_angle_xaxis"].fillna(0).values
            V_theta = plot_df["rotation_angle_yaxis"].fillna(0).values
            W_psi = plot_df["rotation_angle_zaxis"].fillna(0).values

            rotations = R.from_euler("xyz", np.column_stack((U_phi, V_theta, W_psi)), degrees=False) # Assuming radians
            
            # Define an initial reference vector (e.g., along x-axis) to be rotated
            # The choice of this vector depends on how cell orientation is defined.
            # If rotation angles define orientation of a specific cell axis (e.g. its principal dendrite),
            # this vector should represent that axis in the cell's local frame.
            # Common choice: local x-axis [1,0,0] or z-axis [0,0,1] depending on convention.
            ref_vector = np.array([1.0, 0.0, 0.0]) 
            
            # Apply rotations to the reference vector for each cell
            rotated_vectors = rotations.apply(ref_vector)
            
            rot_x, rot_y, rot_z = rotated_vectors[:,0], rotated_vectors[:,1], rotated_vectors[:,2]
            
            label = f"{net_name}: {group_name_val}" if group_key_for_net and group_name_val is not None else net_name
            current_color = cmap(j) if group_key_for_net else "blue"

            h = ax.quiver(
                X, Y, Z, rot_x, rot_y, rot_z,
                color=current_color, label=label,
                arrow_length_ratio=arrow_length_ratio, length=quiver_length # type: ignore
            )
            # Optionally, scatter plot for cell bodies
            # ax.scatter(X, Y, Z, color=current_color, s=10, alpha=0.5) 
            if h not in handles: handles.append(h)


    if not handles:
        print("No data to plot for cell rotations.")
        plt.close(fig)
        return None

    # Auto-scale axes based on plotted data
    all_X = np.concatenate([ax.collections[i].get_offsets()[:,0] for i in range(len(ax.collections))]) if ax.collections else [0]
    all_Y = np.concatenate([ax.collections[i].get_offsets()[:,1] for i in range(len(ax.collections))]) if ax.collections else [0]
    all_Z = np.concatenate([ax.collections[i].get_offsets()[:,2] for i in range(len(ax.collections))]) if ax.collections else [0] # type: ignore
    
    if len(all_X) > 0 : # Check if there is data to plot
        ax.set_xlim([np.min(all_X), np.max(all_X)])
        ax.set_ylim([np.min(all_Y), np.max(all_Y)])
        ax.set_zlim([np.min(all_Z), np.max(all_Z)]) # type: ignore
    
    ax.set_xlabel("X Position (um)")
    ax.set_ylabel("Y Position (um)")
    ax.set_zlabel("Z Position (um)") # type: ignore
    ax.set_title(plot_title)
    
    # Create a legend with unique labels
    unique_labels_handles_rot = {}
    for h_item in handles:
        if h_item.get_label() not in unique_labels_handles_rot:
             unique_labels_handles_rot[h_item.get_label()] = h_item # type: ignore
    if unique_labels_handles_rot:
        ax.legend(handles=unique_labels_handles_rot.values(), loc='center left', bbox_to_anchor=(1.1, 0.5))


    plt.draw()
    if save_file:
        plt.savefig(save_file, dpi=300, bbox_inches="tight")

    if not is_notebook():
        plt.show()
    elif is_notebook() and not save_file:
         plt.show()
    return ax
