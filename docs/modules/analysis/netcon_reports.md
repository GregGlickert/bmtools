# Network Connectivity Synapse Reports

The `netcon_reports` module provides tools for loading and processing detailed synapse-level reports from BMTK simulations. These reports typically contain time-varying data for individual synapses, such as conductance or other biophysical variables.

## Loading Synapse Reports

The primary function in this module is `load_synapse_report`, which reads an HDF5 synapse report file and associated simulation configuration to produce an `xarray.Dataset`. This dataset is structured for easy querying and analysis, with coordinates for time, synapse ID, source/target population, and section-specific information.

## Example Usage

Here's a basic example of how to load a synapse report:

```python
import xarray as xr
from bmtool.analysis.netcon_reports import load_synapse_report

# Define the paths to your simulation output files
# Ensure these paths are correct relative to where your script/notebook is run.
h5_file_path = 'output/synaptic_report.h5' # Example path
config_path = 'config.json'             # Example path
network_name = 'v1_network'             # Example network name used in the simulation

# Load the synapse report
try:
    synapse_ds = load_synapse_report(
        h5_file_path=h5_file_path,
        config_path=config_path,
        network=network_name  # 'network' parameter in the function matches this
    )
    
    # Display the dataset structure
    print("Synapse Dataset:")
    print(synapse_ds)

    # Example: Access data for the first synapse at the first time point
    if 'synapse_value' in synapse_ds and synapse_ds.synapse.size > 0 and synapse_ds.time.size > 0:
        example_value = synapse_ds['synapse_value'].isel(time=0, synapse=0).data
        print(f"\nExample value for synapse 0 at time 0: {example_value}")

    # Example: Get all data for synapses from a specific source population to a target population
    # This requires knowledge of your population names.
    # Replace 'SourcePopName' and 'TargetPopName' with actual names from your model.
    # example_connections = synapse_ds.where(
    #     (synapse_ds.source_pop == 'SourcePopName') & (synapse_ds.target_pop == 'TargetPopName'), 
    #     drop=True
    # )
    # if example_connections.synapse.size > 0:
    #     print("\nData for connections between specific populations:")
    #     print(example_connections)
    # else:
    #     print("\nNo connections found for the example source/target population pair.")

except FileNotFoundError:
    print(f"Error: HDF5 file not found at {h5_file_path} or config not found at {config_path}.")
except Exception as e:
    print(f"An error occurred: {e}")

```

This example demonstrates loading the data. Further analysis would depend on the specific variables stored in your synapse report (e.g., `synapse_value` might represent synaptic conductance, current, etc.) and the scientific questions you are addressing.
The resulting `xarray.Dataset` can be manipulated using xarray's powerful indexing and computation tools.
