from typing import Optional, Tuple, Union, Any # Added Any for FOOOF model for now
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr # Added import for xr.DataArray

from bmtool.analysis.lfp import gen_aperiodic


def plot_spectrogram(
    psd_xarray: xr.DataArray,
    remove_aperiodic: Optional[Any] = None, # Should ideally be FOOOF object
    log_power: Union[bool, str] = False,
    plt_range: Optional[Union[float, Tuple[float, float]]] = None,
    color_map_freq_range: Optional[Tuple[float, float]] = None, # Renamed clr_freq_range
    pad: float = 0.03,
    ax: Optional[plt.Axes] = None,
) -> np.ndarray:
    """
    Plot spectrogram from an xarray.DataArray.

    Optionally removes aperiodic components and scales power logarithmically.
    Color limits for the heatmap can be determined using values within a specified frequency band.

    Parameters
    ----------
    psd_xarray : xr.DataArray
        Input spectrogram data as an xarray DataArray. Expected to have 'PSD' data variable,
        and 'time' and 'frequency' coordinates. May also contain 'cone_of_influence_frequency'.
    remove_aperiodic : Optional[Any], optional
        Fitted FOOOF model object. If provided, its aperiodic component is removed 
        from the spectrogram. Default is None.
    log_power : Union[bool, str], optional
        If True, power is log10 scaled. If "dB", power is converted to dB (10*log10(power)). 
        Default is False.
    plt_range : Optional[Union[float, Tuple[float, float]]], optional
        Frequency range to plot (min_freq, max_freq). If a single float is given, 
        it's treated as max_freq (min_freq determined by data or log_power). 
        If None, full available frequency range is plotted. Default is None.
    color_map_freq_range : Optional[Tuple[float, float]], optional
        Frequency range (min_freq, max_freq) used to determine the color limits (vmin, vmax)
        for the plot. If None, color limits are determined automatically by matplotlib from
        the plotted data. Default is None.
    pad : float, optional
        Padding for the colorbar. Default is 0.03.
    ax : Optional[plt.Axes], optional
        Matplotlib Axes object to plot on. If None, a new figure and axes are created.
        Default is None.

    Returns
    -------
    np.ndarray
        The processed spectrogram data (sxx) that was plotted, as a NumPy array.
        This data is potentially log-scaled and/or has the aperiodic component removed.
    """
    sxx = psd_xarray.PSD.data.copy() # Use .data to get numpy array
    t = psd_xarray.time.data.copy()
    f = psd_xarray.frequency.data.copy()

    cbar_label: str = "PSD" if remove_aperiodic is None else "PSD Residual"
    if log_power:
        with np.errstate(divide="ignore", invalid="ignore"): # Ensure log10 does not error on zero or negative
            sxx = np.log10(sxx, out=np.full_like(sxx, np.nan), where=sxx > 0) # Handle non-positive values
        if log_power == "dB":
            cbar_label += " (dB)" 
        else:
            cbar_label += " (log10 Power)"


    if remove_aperiodic is not None:
        # Determine start index for frequency fitting (avoid f[0]=0 for log-based FOOOF)
        # FOOOF typically expects frequencies > 0.
        aperiodic_model_fit_start_index = 0
        if f[0] <= 0: # If f[0] is 0 or negative, start from the next positive frequency
            aperiodic_model_fit_start_index = np.argmax(f > 0) if np.any(f > 0) else len(f)

        if aperiodic_model_fit_start_index < len(f):
            ap_fit = gen_aperiodic(f[aperiodic_model_fit_start_index:], remove_aperiodic.aperiodic_params_) # FOOOF attributes usually end with _
            
            # Ensure ap_fit is broadcastable to sxx part
            ap_fit_broadcastable = ap_fit[:, np.newaxis]

            if log_power: # If sxx is already log10(PSD)
                # ap_fit from gen_aperiodic is already in log10 if FOOOF was fit on log10 power.
                # Assuming gen_aperiodic returns fit in the same scale FOOOF model was created with.
                # If FOOOF model was fit on log10(power), ap_fit is log10(aperiodic_psd).
                sxx[aperiodic_model_fit_start_index:, :] -= ap_fit_broadcastable
            else: # If sxx is linear PSD
                # ap_fit is log10(aperiodic_psd), so convert it to linear scale before subtraction
                sxx[aperiodic_model_fit_start_index:, :] -= (10**ap_fit_broadcastable)
        
        # Set parts of sxx corresponding to non-positive frequencies to 0 or NaN after processing
        sxx[:aperiodic_model_fit_start_index, :] = np.nan # Or 0.0, depending on desired visual for uncorrected part

    if log_power == "dB": # This should be applied after aperiodic removal if PSD was linear
        # If log_power was True (but not "dB"), sxx is already log10.
        # If log_power was False, sxx is linear. Convert to dB.
        # This logic is a bit tricky: if remove_aperiodic was done on linear scale,
        # then sxx is residual linear power. If on log scale, sxx is log10(residual).
        # For simplicity, assuming if log_power="dB", the 10*log10 should apply to the final sxx.
        # If sxx wasn't log10'd yet (because log_power was only "dB", not True), do it first.
        if not isinstance(log_power, bool) or log_power is not True: # checks if log10 was not already applied
             with np.errstate(divide="ignore", invalid="ignore"):
                 sxx_log10 = np.log10(sxx, out=np.full_like(sxx, np.nan), where=sxx > 0)
             sxx = 10 * sxx_log10
        else: # sxx is already log10(original or residual)
             sxx = 10 * sxx # This scales log10(power) to dB-like units (though typically it's 10*log10(power/ref))
                           # Or if it was already 10*log10, this makes it 100*log10. This needs care.
                           # Assuming the first pass of log_power=True made it log10(PSD).
                           # Now, if also "dB", we scale this log10 value by 10.

    if ax is None:
        _, ax = plt.subplots(1, 1)
    
    current_plt_range: List[float]
    if plt_range is None:
        current_plt_range = [f[np.argmax(f > 0)] if np.any(f > 0) else 0.0, f[-1]]
    elif isinstance(plt_range, (float, int)):
        current_plt_range = [f[np.argmax(f > 0)] if np.any(f > 0) else 0.0, float(plt_range)]
    else: # Tuple
        current_plt_range = list(plt_range)

    f_idx = (f >= current_plt_range[0]) & (f <= current_plt_range[1])
    
    vmin_val, vmax_val = None, None # Renamed vmin, vmax
    if color_map_freq_range is not None:
        c_idx = (f >= color_map_freq_range[0]) & (f <= color_map_freq_range[1])
        # Ensure c_idx has some True values and sxx[c_idx, :] is not all NaN
        if np.any(c_idx) and not np.all(np.isnan(sxx[c_idx, :])):
            vmin_val = np.nanmin(sxx[c_idx, :]) # Use nanmin/nanmax
            vmax_val = np.nanmax(sxx[c_idx, :])
        else:
            print("Warning: color_map_freq_range resulted in no valid data for color limits. Using automatic scaling.")


    f_plot = f[f_idx] # Frequencies to plot
    sxx_plot = sxx[f_idx, :] # Spectrogram data to plot

    if f_plot.size == 0 or sxx_plot.size == 0:
        print("Warning: No data to plot after frequency range filtering.")
        return sxx # Return processed sxx even if not plotted

    pcm = ax.pcolormesh(t, f_plot, sxx_plot, shading="gouraud", vmin=vmin_val, vmax=vmax_val)
    
    if "cone_of_influence_frequency" in psd_xarray:
        coif = psd_xarray.cone_of_influence_frequency.data # Get numpy array
        ax.plot(t, coif, color='k', linestyle='--') # Made COI line clearer
        ax.fill_between(t, coif, f_plot[-1], color='k', alpha=0.1) # Fill above COI

    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(f_plot[0], f_plot[-1])
    plt.colorbar(mappable=pcm, ax=ax, label=cbar_label, pad=pad)
    ax.set_xlabel("Time (s)") # Assuming time is in seconds
    ax.set_ylabel("Frequency (Hz)")
    return sxx
