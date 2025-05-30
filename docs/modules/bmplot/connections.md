# Network Connections Plotting

The `bmtool.bmplot.connections` module provides a suite of functions for visualizing various aspects of network connectivity, connection properties, cell positions, and orientations from BMTK simulations. These plots help in understanding the structure and potential communication pathways within the simulated neural networks.

## Visualizing Connection Matrices

Connection matrices provide a high-level overview of how different populations or cell groups connect to each other.

### Total Connections or Synaptic Properties

You can visualize the total number of connections, mean/std of connections, or types of synapses (.mod or .json files) between source and target populations.

```python
from bmtool.bmplot.connections import total_connection_matrix
import matplotlib.pyplot as plt

# Assuming 'config.json' exists and specifies networks like 'network_A' and 'network_B'
# This will plot the total number of connections from source populations in network_A
# to target populations in network_B.
total_connection_matrix(
    config="config.json",
    sources="network_A",       # Comma-separated source network names
    targets="network_B",       # Comma-separated target network names
    title="Total Connections: Network A to Network B",
    synaptic_info="0"          # '0' for total connections
    # save_file="total_connections.png" # Optional: to save the figure
)
# plt.show() # Typically not needed if running in a Jupyter notebook that auto-displays plots
```

### Percentage Connectivity

To visualize the percentage of connections:

```python
from bmtool.bmplot.connections import percent_connection_matrix

percent_connection_matrix(
    config="config.json",
    sources="network_A",
    targets="network_B",
    title="Percentage Connectivity: Network A to Network B",
    method="total",  # Can be 'total', 'uni', or 'bi'
    include_gap=True
    # save_file="percent_connectivity.png" # Optional
)
# plt.show()
```

### Convergence/Divergence Statistics

Plotting convergence (number of inputs to target cells) or divergence (number of outputs from source cells):

```python
from bmtool.bmplot.connections import convergence_connection_matrix # or divergence_connection_matrix

# Example for convergence
convergence_connection_matrix(
    config="config.json",
    sources="network_A",
    targets="network_B",
    title="Mean Synaptic Convergence on Network B from Network A",
    method="mean",  # Options: 'mean', 'min', 'max', 'std', 'mean+std'
    # return_dict=True # Set to True to get data as dict instead of plotting
    # save_file="convergence_mean.png" # Optional
)
# plt.show()
```

## Distribution of Edge Properties

Visualize the distribution of specific synaptic properties (e.g., weights, delays) as a matrix of histograms.

```python
from bmtool.bmplot.connections import edge_histogram_matrix

edge_histogram_matrix(
    config="config.json",
    sources="network_A",
    targets="network_B",
    edge_property="syn_weight", # Replace with the desired edge property
    title="Synaptic Weight Distribution: Network A to Network B",
    # report="some_report_name" # If edge properties are in a specific report
    # save_file="edge_prop_hist.png" # Optional
)
# plt.show()
```

## Spatial Visualization

### 3D Cell Positions

Plot the physical locations of cells in 3D space.

```python
from bmtool.bmplot.connections import plot_3d_positions

plot_3d_positions(
    config="config.json",
    sources="network_A,network_B", # Plot cells from both networks
    sid="pop_name",                # Color/group cells by their 'pop_name'
    title="3D Cell Positions by Population",
    subset=10                      # Plot every 10th cell to keep plot clean for large networks
    # save_file="cell_positions.png" # Optional
)
# plt.show()
```

### 3D Cell Rotations (Orientations)

Visualize the orientation of cells in 3D space using quiver plots.

```python
from bmtool.bmplot.connections import plot_3d_cell_rotation

plot_3d_cell_rotation(
    config="config.json",
    sources="network_A",
    sids="pop_name", # Group by population name for coloring vectors
    title="Cell Orientations in Network A",
    quiver_length=10.0,      # Adjust based on your model's scale
    arrow_length_ratio=0.3
    # save_file="cell_rotations.png" # Optional
)
# plt.show()
```

These examples demonstrate some of the key plotting capabilities for network connections and cell properties. Refer to the specific function docstrings in the [API Documentation](../api/bmplot/connections.md) for more details on parameters and options.
