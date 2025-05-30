import h5py
import numpy as np
from typing import Dict, List # Add typing imports
import pandas as pd # Add pandas import for DataFrame type hint

import xarray as xr

from ..util.util import load_nodes_from_config


def load_synapse_report(h5_file_path: str, config_path: str, network: str) -> xr.Dataset:
    """
    Load and process a synapse report from a bmtk simulation into an xarray.Dataset.

    Parameters
    ----------
    h5_file_path : str
        Path to the HDF5 file containing the synapse report.
    config_path : str
        Path to the simulation configuration file (e.g., config.json).
    network : str
        The name of the network within the report (e.g., 'v1', 'lgn').

    Returns
    -------
    xr.Dataset
        An xarray Dataset containing the synapse report data with dimensions
        ('time', 'synapse') and various coordinates for synapse properties
        and population labeling.
    """
    # Load the h5 file
    with h5py.File(h5_file_path, "r") as file:
        # Get the report data
        report = file["report"][network]
        mapping = report["mapping"]

        # Get the data - shape is (n_timesteps, n_synapses)
        data = report["data"][:]

        # Get time information
        time_info = mapping["time"][:]  # [start_time, end_time, dt]
        start_time = time_info[0]
        end_time = time_info[1]
        dt = time_info[2]

        # Create time array
        n_steps = data.shape[0]
        time = np.linspace(start_time, start_time + (n_steps - 1) * dt, n_steps)

        # Get mapping information
        src_ids = mapping["src_ids"][:]
        trg_ids = mapping["trg_ids"][:]
        sec_id = mapping["element_ids"][:]
        sec_x = mapping["element_pos"][:]

    # Load node information
    nodes_df: pd.DataFrame = load_nodes_from_config(config_path) # type: ignore
    # The load_nodes_from_config likely returns a dict of DataFrames, 
    # so we select the one for the specified network.
    network_nodes: pd.DataFrame = nodes_df[network] # type: ignore

    # Create a mapping from node IDs to population names
    node_to_pop: Dict[int, str] = dict(zip(network_nodes.index, network_nodes["pop_name"]))

    # Get the number of synapses
    n_synapses: int = data.shape[1]

    # Create arrays to hold the source and target populations for each synapse
    source_pops: List[str] = []
    target_pops: List[str] = []
    connection_labels: List[str] = []

    # Process each synapse
    for i in range(n_synapses):
        src_id = src_ids[i]
        trg_id = trg_ids[i]

        # Get population names (with fallback for unknown IDs)
        src_pop = node_to_pop.get(src_id, f"unknown_{src_id}")
        trg_pop = node_to_pop.get(trg_id, f"unknown_{trg_id}")

        source_pops.append(src_pop)
        target_pops.append(trg_pop)
        connection_labels.append(f"{src_pop}->{trg_pop}")

    # Create xarray dataset
    ds = xr.Dataset(
        data_vars={"synapse_value": (["time", "synapse"], data)},
        coords={
            "time": time,
            "synapse": np.arange(n_synapses),
            "source_pop": ("synapse", source_pops),
            "target_pop": ("synapse", target_pops),
            "source_id": ("synapse", src_ids),
            "target_id": ("synapse", trg_ids),
            "sec_id": ("synapse", sec_id),
            "sec_x": ("synapse", sec_x),
            "connection_label": ("synapse", connection_labels),
        },
        attrs={"description": "Synapse report data from bmtk simulation"},
    )

    return ds
