import numpy as np
import plotly.graph_objects as go
from data_handling import Analysis_PSWS

import numpy as np
import matplotlib.pyplot as plt


def plot_tof_comparison_old(
    # ---------------------------------------------------------
    # Device 1
    # ---------------------------------------------------------
    t1,
    signal_t1,
    gated_t1=None,
    gates_t1=None,
    freq1=None,
    gated_f1=None,

    # ---------------------------------------------------------
    # Device 2
    # ---------------------------------------------------------
    t2=None,
    signal_t2=None,
    gated_t2=None,
    gates_t2=None,
    freq2=None,
    gated_f2=None,

    # ---------------------------------------------------------
    # Axis limits
    # ---------------------------------------------------------
    time_xlim=None,
    time_ylim=None,
    freq_xlim=None,
    freq_ylim=None,

    # ---------------------------------------------------------
    # Labels
    # ---------------------------------------------------------
    device1_label="Device 1",
    device2_label="Device 2",

    # ---------------------------------------------------------
    # Plot appearance
    # ---------------------------------------------------------
    figsize=(7.2, 7.0),
    linewidth=1.5,
    gate_alpha=0.15,
    fontsize=9,

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------
    save_path=None,
):
    """
    Publication-ready 3 x 2 comparison plot for time-of-flight data.

    Rows
    ----
    Row 1 : Original time-domain signal + gate regions
    Row 2 : Time-domain gated signals
    Row 3 : Frequency-domain gated signals

    Columns
    -------
    Column 1 : Device 1
    Column 2 : Device 2

    Parameters
    ----------
    t1, t2 : arrays
        Time axes in ns.

    signal_t1, signal_t2 : arrays
        Original complex time-domain signals.

    gated_t1, gated_t2 : list of arrays or None
        Gated complex time-domain signals.
        Each list can contain 0, 1, or 2 signals.

    gates_t1, gates_t2 : list of arrays or None
        Corresponding gate arrays.
        Each list can contain 0, 1, or 2 gates.

    freq1, freq2 : arrays
        Frequency axes in GHz.

    gated_f1, gated_f2 : list of arrays or None
        Gated complex frequency-domain signals.
        Each list can contain 0, 1, or 2 spectra.

    time_xlim : tuple or None
        Example: (0, 20)

    time_ylim : tuple or None
        Example: (0, 0.015)

    freq_xlim : tuple or None
        Example: (0, 6)

    freq_ylim : tuple or None
        Example: (0, 0.01)

    device1_label, device2_label : str
        Column titles.

    figsize : tuple
        Figure size in inches.

    linewidth : float
        Main trace linewidth.

    gate_alpha : float
        Transparency of gate regions.

    fontsize : float
        Base font size.

    save_path : str or None
        If provided, save directly as PDF.
    """

    # =========================================================
    # Prepare inputs
    # =========================================================

    if gated_t1 is None:
        gated_t1 = []

    if gated_t2 is None:
        gated_t2 = []

    if gates_t1 is None:
        gates_t1 = []

    if gates_t2 is None:
        gates_t2 = []

    if gated_f1 is None:
        gated_f1 = []

    if gated_f2 is None:
        gated_f2 = []

    if gates_t1 and len(gates_t1) > 2:
        raise ValueError("Maximum of two gates allowed for Device 1.")

    if gates_t2 and len(gates_t2) > 2:
        raise ValueError("Maximum of two gates allowed for Device 2.")

    if len(gated_t1) > 2:
        raise ValueError("Maximum of two gated signals allowed for Device 1.")

    if len(gated_t2) > 2:
        raise ValueError("Maximum of two gated signals allowed for Device 2.")

    if len(gated_f1) > 2:
        raise ValueError("Maximum of two gated spectra allowed for Device 1.")

    if len(gated_f2) > 2:
        raise ValueError("Maximum of two gated spectra allowed for Device 2.")

    # =========================================================
    # Colors
    # =========================================================

    # Muted publication-style colors
    colors = [
        "#0072B2",   # dark blue
        "#D55E00",   # vermillion
    ]

    original_color = "#222222"

    # =========================================================
    # Figure
    # =========================================================

    fig, axes = plt.subplots(
        3,
        2,
        figsize=figsize,
        sharex=False,
    )

    ax11, ax12 = axes[0]
    ax21, ax22 = axes[1]
    ax31, ax32 = axes[2]

    # =========================================================
    # Helper: format axes
    # =========================================================

    def format_axis(ax):

        ax.tick_params(
            direction="in",
            which="both",
            top=True,
            right=True,
            labelsize=fontsize,
        )

        ax.tick_params(
            which="major",
            length=4,
            width=0.8,
        )

        ax.tick_params(
            which="minor",
            length=2,
            width=0.6,
        )

        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    # =========================================================
    # Helper: plot gates on original time trace
    # =========================================================

    def plot_gates(ax, t, gates):

        for i, gate in enumerate(gates):

            if gate is None:
                continue

            gate = np.asarray(gate)

            # Find region where gate is non-zero
            indices = np.where(gate > 1e-6)[0]

            if len(indices) == 0:
                continue

            t_start = t[indices[0]]
            t_stop = t[indices[-1]]

            ax.axvspan(
                t_start,
                t_stop,
                color=colors[i],
                alpha=gate_alpha,
                linewidth=0,
            )

            # Gate boundaries
            ax.axvline(
                t_start,
                color=colors[i],
                linestyle="--",
                linewidth=0.8,
                alpha=0.7,
            )

            ax.axvline(
                t_stop,
                color=colors[i],
                linestyle="--",
                linewidth=0.8,
                alpha=0.7,
            )

    # =========================================================
    # DEVICE 1 — ROW 1
    # =========================================================

    ax11.plot(
        t1,
        np.abs(signal_t1),
        color=original_color,
        linewidth=linewidth,
    )

    plot_gates(
        ax11,
        t1,
        gates_t1,
    )

    ax11.set_ylabel(
        r"$|S_{21}(t)|$",
        fontsize=fontsize,
    )

    ax11.set_title(
        device1_label,
        fontsize=fontsize + 1,
        pad=6,
    )

    # =========================================================
    # DEVICE 2 — ROW 1
    # =========================================================

    if t2 is not None and signal_t2 is not None:

        ax12.plot(
            t2,
            np.abs(signal_t2),
            color=original_color,
            linewidth=linewidth,
        )

        plot_gates(
            ax12,
            t2,
            gates_t2,
        )

    ax12.set_title(
        device2_label,
        fontsize=fontsize + 1,
        pad=6,
    )

    # =========================================================
    # DEVICE 1 — ROW 2
    # =========================================================

    for i, signal in enumerate(gated_t1):

        ax21.plot(
            t1,
            np.abs(signal),
            color=colors[i],
            linewidth=linewidth,
            label=f"Gate {i + 1}",
        )

    ax21.set_ylabel(
        r"$|S_{21}^{\mathrm{gate}}(t)|$",
        fontsize=fontsize,
    )

    if len(gated_t1) > 0:
        ax21.legend(
            frameon=False,
            fontsize=fontsize - 1,
            loc="best",
        )

    # =========================================================
    # DEVICE 2 — ROW 2
    # =========================================================

    if t2 is not None:

        for i, signal in enumerate(gated_t2):

            ax22.plot(
                t2,
                np.abs(signal),
                color=colors[i],
                linewidth=linewidth,
                label=f"Gate {i + 1}",
            )

    if len(gated_t2) > 0:
        ax22.legend(
            frameon=False,
            fontsize=fontsize - 1,
            loc="best",
        )

    # =========================================================
    # DEVICE 1 — ROW 3
    # =========================================================

    for i, spectrum in enumerate(gated_f1):

        ax31.plot(
            freq1,
            np.abs(spectrum),
            color=colors[i],
            linewidth=linewidth,
            label=f"Gate {i + 1}",
        )

    ax31.set_xlabel(
        "Frequency (GHz)",
        fontsize=fontsize,
    )

    ax31.set_ylabel(
        r"$|S_{21}^{\mathrm{gate}}(f)|$",
        fontsize=fontsize,
    )

    # =========================================================
    # DEVICE 2 — ROW 3
    # =========================================================

    if freq2 is not None:

        for i, spectrum in enumerate(gated_f2):

            ax32.plot(
                freq2,
                np.abs(spectrum),
                color=colors[i],
                linewidth=linewidth,
                label=f"Gate {i + 1}",
            )

    ax32.set_xlabel(
        "Frequency (GHz)",
        fontsize=fontsize,
    )

    # =========================================================
    # Axis limits
    # =========================================================

    for ax in [ax11, ax12, ax21, ax22]:

        if time_xlim is not None:
            ax.set_xlim(time_xlim)

        if time_ylim is not None:
            ax.set_ylim(time_ylim)

    for ax in [ax31, ax32]:

        if freq_xlim is not None:
            ax.set_xlim(freq_xlim)

        if freq_ylim is not None:
            ax.set_ylim(freq_ylim)

    # =========================================================
    # Row labels / panel labels
    # =========================================================

    ax11.text(
        -0.15,
        1.02,
        "(a)",
        transform=ax11.transAxes,
        fontsize=fontsize,
        fontweight="bold",
    )

    ax12.text(
        -0.08,
        1.02,
        "(b)",
        transform=ax12.transAxes,
        fontsize=fontsize,
        fontweight="bold",
    )

    ax21.text(
        -0.15,
        1.02,
        "(c)",
        transform=ax21.transAxes,
        fontsize=fontsize,
        fontweight="bold",
    )

    ax22.text(
        -0.08,
        1.02,
        "(d)",
        transform=ax22.transAxes,
        fontsize=fontsize,
        fontweight="bold",
    )

    ax31.text(
        -0.15,
        1.02,
        "(e)",
        transform=ax31.transAxes,
        fontsize=fontsize,
        fontweight="bold",
    )

    ax32.text(
        -0.08,
        1.02,
        "(f)",
        transform=ax32.transAxes,
        fontsize=fontsize,
        fontweight="bold",
    )

    # =========================================================
    # Format all axes
    # =========================================================

    for ax in axes.flat:
        format_axis(ax)

    # =========================================================
    # Layout
    # =========================================================

    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        bottom=0.08,
        top=0.95,
        wspace=0.22,
        hspace=0.28,
    )

    # =========================================================
    # Save PDF
    # =========================================================

    if save_path is not None:

        fig.savefig(
            save_path,
            format="pdf",
            bbox_inches="tight",
        )

    return fig, axes

def plot_tof_comparison(
    # Device 1
    t1,
    signal_t1,
    gated_t1=None,
    gates_t1=None,
    freq1=None,
    gated_f1=None,

    # Device 2
    t2=None,
    signal_t2=None,
    gated_t2=None,
    gates_t2=None,
    freq2=None,
    gated_f2=None,

    # X limits
    time_xlim=None,
    freq_xlim=None,

    # Independent Y limits
    time_ylim1=None,
    time_ylim2=None,
    freq_ylim1=None,
    freq_ylim2=None,

    # Y-axis scaling
    y_scale=1,

    # Labels
    device1_label="Device 1",
    device2_label="Device 2",

    # Saving
    save_path=None,

    # Figure
    figsize=(7.2, 7.0),
):
    """
    Publication-ready comparison plot for time-of-flight spectroscopy.

    Layout:
        Row 1: Original time-domain FT + gate regions
        Row 2: Gated time-domain signals
        Row 3: Frequency-domain spectra after gating

        Column 1: Device 1
        Column 2: Device 2

    Parameters
    ----------
    y_scale : float
        Scale factor used to make small signals easier to read.

        Example:
            y_scale = 1e-3

        means the plotted quantity is:

            signal / 1e-3

        so a signal of 0.002 appears as 2.

        The y-axis label will indicate the scaling.

    time_ylim1, time_ylim2 : tuple or None
        Independent y-axis limits for row 1 and 2 for each device.

        time_ylim1 applies to Device 1.
        time_ylim2 applies to Device 2.

        Note:
        The same limit is used for rows 1 and 2 for a given device.

    freq_ylim1, freq_ylim2 : tuple or None
        Independent frequency-domain y-axis limits for each device.

    gated_t1, gated_t2 : list
        Lists containing 0, 1, or 2 gated time-domain signals.

    gates_t1, gates_t2 : list
        Lists containing the corresponding gate arrays.

    gated_f1, gated_f2 : list
        Lists containing the corresponding reconstructed frequency-domain
        spectra.

    freq1, freq2 : array
        Frequency axes corresponding to gated_f1 and gated_f2.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    # ------------------------------------------------------------
    # Defaults
    # ------------------------------------------------------------

    if gated_t1 is None:
        gated_t1 = []

    if gated_t2 is None:
        gated_t2 = []

    if gates_t1 is None:
        gates_t1 = []

    if gates_t2 is None:
        gates_t2 = []

    if gated_f1 is None:
        gated_f1 = []

    if gated_f2 is None:
        gated_f2 = []

    # ------------------------------------------------------------
    # Colors
    # ------------------------------------------------------------

    # Original FT: dark blue instead of black
    original_color = "#3A506B"

    # Gate / gated-signal colors
    gate_colors = [
        "#0072B2",   # dark blue
        "#D55E00",   # vermillion
    ]

    # ------------------------------------------------------------
    # Scaling
    # ------------------------------------------------------------

    if y_scale <= 0:
        raise ValueError("y_scale must be positive.")

    # Construct scaling label
    exponent = int(np.floor(np.log10(y_scale)))

    # Only use scientific notation when scaling is actually useful
    if not np.isclose(y_scale, 1):
        exponent = int(np.round(-np.log10(y_scale)))
        scale_label = rf"$|S_{{21}}| \times 10^{{{exponent}}}$"
    else:
        scale_label = r"$|S_{21}|$"

    # ------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------

    def scaled_abs(signal):
        """Return absolute value divided by y_scale."""
        return np.abs(signal) / y_scale

    def setup_axis(ax):
        """General publication-style axis formatting."""
        ax.tick_params(
            direction="in",
            which="both",
            top=True,
            right=True,
            labelsize=9,
        )

        ax.minorticks_on()

        ax.tick_params(
            which="minor",
            length=3,
        )

        ax.tick_params(
            which="major",
            length=5,
        )

    def add_gate_to_axis(ax, t, gate, color, label):
        """
        Add a shaded gate region and dashed boundaries to an axis.
        """
        gate = np.asarray(gate)

        if len(gate) == 0:
            return

        active = np.where(gate > 1e-6)[0]

        if len(active) == 0:
            return

        t_start = t[active[0]]
        t_stop = t[active[-1]]

        # Shaded gate region
        ax.axvspan(
            t_start,
            t_stop,
            color=color,
            alpha=0.12,
            lw=0,
        )

        # Gate boundaries
        ax.axvline(
            t_start,
            color=color,
            linestyle="--",
            linewidth=0.9,
            alpha=0.8,
        )

        ax.axvline(
            t_stop,
            color=color,
            linestyle="--",
            linewidth=0.9,
            alpha=0.8,
        )

        # Dummy line for legend
        ax.plot(
            [],
            [],
            color=color,
            linestyle="--",
            linewidth=1.2,
            label=label,
        )

    def format_scientific_axis(ax):
        """
        Use scientific notation on the y axis when appropriate.
        """
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 2))
        ax.yaxis.set_major_formatter(formatter)

    # ------------------------------------------------------------
    # Create figure
    # ------------------------------------------------------------

    fig, axes = plt.subplots(
        3,
        2,
        figsize=figsize,
        sharex=False,
    )

    ax11 = axes[0, 0]
    ax12 = axes[0, 1]

    ax21 = axes[1, 0]
    ax22 = axes[1, 1]

    ax31 = axes[2, 0]
    ax32 = axes[2, 1]

    # ------------------------------------------------------------
    # ROW 1
    # Original time-domain FT
    # ------------------------------------------------------------

    ax11.plot(
        t1,
        scaled_abs(signal_t1),
        color=original_color,
        linewidth=1.3,
        label="Original FT",
    )

    ax12.plot(
        t2,
        scaled_abs(signal_t2),
        color=original_color,
        linewidth=1.3,
        label="Original FT",
    )

    # Add gates on top of original FT
    for i, gate in enumerate(gates_t1[:2]):
        add_gate_to_axis(
            ax11,
            t1,
            gate,
            gate_colors[i],
            f"Gate {i + 1}",
        )

    for i, gate in enumerate(gates_t2[:2]):
        add_gate_to_axis(
            ax12,
            t2,
            gate,
            gate_colors[i],
            f"Gate {i + 1}",
        )

    ax11.set_ylabel(scale_label, fontsize=10)
    ax12.set_ylabel(scale_label, fontsize=10)

    ax11.set_title(device1_label, fontsize=11)
    ax12.set_title(device2_label, fontsize=11)

    # Legends for gates
    ax11.legend(
        loc="best",
        fontsize=8,
        frameon=False,
    )

    ax12.legend(
        loc="best",
        fontsize=8,
        frameon=False,
    )

    # ------------------------------------------------------------
    # ROW 2
    # Gated time-domain signals
    # ------------------------------------------------------------

    for i, signal in enumerate(gated_t1[:2]):

        if signal is None or len(signal) == 0:
            continue

        ax21.plot(
            t1,
            scaled_abs(signal),
            color=gate_colors[i],
            linewidth=1.3,
            label=f"Gated signal {i + 1}",
        )

    for i, signal in enumerate(gated_t2[:2]):

        if signal is None or len(signal) == 0:
            continue

        ax22.plot(
            t2,
            scaled_abs(signal),
            color=gate_colors[i],
            linewidth=1.3,
            label=f"Gated signal {i + 1}",
        )

    ax21.set_ylabel(scale_label, fontsize=10)
    ax22.set_ylabel(scale_label, fontsize=10)

    # Correct legends: Gated signal 1 / 2
    if len(gated_t1) > 0:
        ax21.legend(
            loc="best",
            fontsize=8,
            frameon=False,
        )

    if len(gated_t2) > 0:
        ax22.legend(
            loc="best",
            fontsize=8,
            frameon=False,
        )

    # ------------------------------------------------------------
    # ROW 3
    # Reconstructed frequency-domain spectra
    # ------------------------------------------------------------

    for i, spectrum in enumerate(gated_f1[:2]):

        if spectrum is None or len(spectrum) == 0:
            continue

        if freq1 is None:
            raise ValueError(
                "freq1 must be provided when gated_f1 is used."
            )

        if len(freq1) != len(spectrum):
            raise ValueError(
                f"Device 1: freq1 has length {len(freq1)}, "
                f"but gated_f1[{i}] has length {len(spectrum)}."
            )

        ax31.plot(
            freq1,
            scaled_abs(spectrum),
            color=gate_colors[i],
            linewidth=1.3,
            label=f"Gated signal {i + 1}",
        )

    for i, spectrum in enumerate(gated_f2[:2]):

        if spectrum is None or len(spectrum) == 0:
            continue

        if freq2 is None:
            raise ValueError(
                "freq2 must be provided when gated_f2 is used."
            )

        if len(freq2) != len(spectrum):
            raise ValueError(
                f"Device 2: freq2 has length {len(freq2)}, "
                f"but gated_f2[{i}] has length {len(spectrum)}."
            )

        ax32.plot(
            freq2,
            scaled_abs(spectrum),
            color=gate_colors[i],
            linewidth=1.3,
            label=f"Gated signal {i + 1}",
        )

    ax31.set_xlabel("Frequency (GHz)", fontsize=10)
    ax32.set_xlabel("Frequency (GHz)", fontsize=10)

    ax31.set_ylabel(scale_label, fontsize=10)
    ax32.set_ylabel(scale_label, fontsize=10)

    if len(gated_f1) > 0:
        ax31.legend(
            loc="best",
            fontsize=8,
            frameon=False,
        )

    if len(gated_f2) > 0:
        ax32.legend(
            loc="best",
            fontsize=8,
            frameon=False,
        )

    # ------------------------------------------------------------
    # Axis limits
    # ------------------------------------------------------------

    # X limits: same within each row/device
    if time_xlim is not None:
        ax11.set_xlim(time_xlim)
        ax12.set_xlim(time_xlim)
        ax21.set_xlim(time_xlim)
        ax22.set_xlim(time_xlim)

    if freq_xlim is not None:
        ax31.set_xlim(freq_xlim)
        ax32.set_xlim(freq_xlim)

    # Independent Y limits
    if time_ylim1 is not None:
        ax11.set_ylim(time_ylim1)
        ax21.set_ylim(time_ylim1)

    if time_ylim2 is not None:
        ax12.set_ylim(time_ylim2)
        ax22.set_ylim(time_ylim2)

    if freq_ylim1 is not None:
        ax31.set_ylim(freq_ylim1)

    if freq_ylim2 is not None:
        ax32.set_ylim(freq_ylim2)

    # ------------------------------------------------------------
    # Axis formatting
    # ------------------------------------------------------------

    for ax in axes.flat:
        setup_axis(ax)

    # Scientific notation formatting
    for ax in axes.flat:
        format_scientific_axis(ax)

    # ------------------------------------------------------------
    # Hide redundant x tick labels
    # ------------------------------------------------------------

    ax11.tick_params(labelbottom=False)
    ax12.tick_params(labelbottom=False)

    ax21.tick_params(labelbottom=False)
    ax22.tick_params(labelbottom=False)

    # ------------------------------------------------------------
    # Make columns visually independent
    # ------------------------------------------------------------

    # Keep y-axis labels on both columns because the scales can differ
    ax12.yaxis.set_label_position("left")
    ax22.yaxis.set_label_position("left")
    ax32.yaxis.set_label_position("left")

    # ------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------

    fig.subplots_adjust(
        left=0.11,
        right=0.98,
        bottom=0.09,
        top=0.95,
        wspace=0.28,
        hspace=0.28,
    )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    if save_path is not None:
        fig.savefig(
            save_path,
            bbox_inches="tight",
        )

    return fig, axes


def plot_tof_comparison_claude(
    # Device 1
    t1,
    signal_t1,
    gated_t1=None,
    gates_t1=None,
    freq1=None,
    gated_f1=None,
    freq_raw1=None,      # original frequency axis (Device 1)
    raw_f1=None,         # original S trace (Device 1)

    # Device 2
    t2=None,
    signal_t2=None,
    gated_t2=None,
    gates_t2=None,
    freq2=None,
    gated_f2=None,
    freq_raw2=None,      # original frequency axis (Device 2)
    raw_f2=None,         # original S trace (Device 2)

    # X limits
    time_xlim=None,
    freq_xlim=None,

    # Independent Y limits
    time_ylim1=None,
    time_ylim2=None,
    freq_ylim1=None,
    freq_ylim2=None,

    # Y-axis scaling
    y_scale=1,

    # Labels
    device1_label="Device 1",
    device2_label="Device 2",
    time_label="Time (ns)",

    # Font sizes
    label_fs=10,         # axis labels (x and y)
    tick_fs=9,           # tick numbers (and the x10^n offset text)
    title_fs=11,         # device titles
    legend_fs=8,         # legends

    # Layout
    time_hspace=0.12,    # gap between row 1 and row 2

    # Saving
    save_path=None,

    # Figure
    figsize=(7.2, 7.0),
):
    """
    Publication-ready comparison plot for time-of-flight spectroscopy.

    Layout (columns: Device 1 | Device 2):
        Row 1: Original time-domain FT + gate regions   \  share one
        Row 2: Gated time-domain signals                /  time axis
        Row 3: Raw frequency-domain data (grey, background) with the
               gated, reconstructed spectra on top

    y_scale : float
        Plotted quantity is |signal| / y_scale (e.g. 1e-3 -> label x10^3).
    time_ylim1/2, freq_ylim1/2 : tuple or None
        Independent y limits per device. The same time_ylim is used for
        rows 1 and 2 of a given device.
    gated_t*, gates_t*, gated_f* : lists with 0, 1 or 2 arrays.
    freq_raw*, raw_f* : original frequency axis and original S trace
        (optional). Plotted in grey behind the gated spectra in row 3.
    label_fs, tick_fs, title_fs, legend_fs : float
        Font sizes (pt) for axis labels, tick numbers, device titles
        and legends.
    time_hspace : float
        Vertical gap between rows 1 and 2 (fraction of axis height).
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    from matplotlib.ticker import ScalarFormatter

    # ------------------------------------------------------------
    # Defaults
    # ------------------------------------------------------------
    gated_t1 = [] if gated_t1 is None else gated_t1
    gated_t2 = [] if gated_t2 is None else gated_t2
    gates_t1 = [] if gates_t1 is None else gates_t1
    gates_t2 = [] if gates_t2 is None else gates_t2
    gated_f1 = [] if gated_f1 is None else gated_f1
    gated_f2 = [] if gated_f2 is None else gated_f2

    if t2 is None or signal_t2 is None:
        raise ValueError("t2 and signal_t2 must be provided for Device 2.")

    # ------------------------------------------------------------
    # Colors
    # ------------------------------------------------------------
    original_color = "#3A506B"
    gate_colors = ["#0072B2", "#D55E00"]
    raw_color = "#D9A044"#"#909090"#"#B4B4B4"   # light grey for the background trace

    # ------------------------------------------------------------
    # Scaling
    # ------------------------------------------------------------
    if y_scale <= 0:
        raise ValueError("y_scale must be positive.")

    if not np.isclose(y_scale, 1):
        exponent = int(np.round(-np.log10(y_scale)))
        scale_suffix = rf"(\times 10^{{{-exponent}}} U)"
    else:
        scale_suffix = ""

    def make_ylabel(core):
        """Build a mathtext y label, appending the scale factor if any."""
        return rf"$|{core}|{scale_suffix}$"

    ylabel_row1 = make_ylabel(r"\mathcal{FT}\;[S_{21}]")
    ylabel_row2 = make_ylabel(r"\mathcal{FT}\;[S_{21}]")
    ylabel_row3 = make_ylabel(r"S_{21}")

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    def scaled_abs(signal):
        return np.abs(signal) / y_scale

    def setup_axis(ax):
        ax.tick_params(direction="in", which="both",
                       top=True, right=True, labelsize=tick_fs)
        ax.minorticks_on()
        ax.tick_params(which="minor", length=3)
        ax.tick_params(which="major", length=5)

    def add_gate_to_axis(ax, t, gate, color, label):
        gate = np.asarray(gate)
        if len(gate) == 0:
            return
        active = np.where(gate > 1e-6)[0]
        if len(active) == 0:
            return

        t_start = t[active[0]]
        t_stop = t[active[-1]]

        ax.axvspan(t_start, t_stop, color=color, alpha=0.12, lw=0)
        for edge in (t_start, t_stop):
            ax.axvline(edge, color=color, linestyle="--",
                       linewidth=0.9, alpha=0.8)

        # Dummy line for legend
        ax.plot([], [], color=color, linestyle="--",
                linewidth=1.2, label=label)

    def format_scientific_axis(ax):
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 2))
        ax.yaxis.set_major_formatter(formatter)
        # the "x10^n" offset text at the top of the y axis
        ax.yaxis.get_offset_text().set_fontsize(tick_fs)

    def plot_raw(ax, f_raw, s_raw, device_name):
        """Grey background trace with the original frequency data."""
        if s_raw is None or len(s_raw) == 0:
            return False
        if f_raw is None:
            raise ValueError(
                f"{device_name}: the raw frequency axis must be provided "
                f"when the raw trace is used."
            )
        if len(f_raw) != len(s_raw):
            raise ValueError(
                f"{device_name}: raw frequency axis has length {len(f_raw)}, "
                f"but raw trace has length {len(s_raw)}."
            )
        ax.plot(f_raw, scaled_abs(s_raw), color=raw_color,
                linewidth=3.5, zorder=1, label="Raw data")
        return True

    # ------------------------------------------------------------
    # Figure and grid
    # ------------------------------------------------------------
    fig = plt.figure(figsize=figsize)

    # Outer grid: [time block (rows 1+2)] / [frequency row (row 3)]
    outer = GridSpec(
        2, 2, figure=fig,
        height_ratios=[2.0, 1.0],
        left=0.11, right=0.98, bottom=0.08, top=0.95,
        wspace=0.22, hspace=0.22,
    )

    # Inner grids for the time block: small gap between rows 1 and 2
    inner_left = GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[0, 0], hspace=time_hspace)
    inner_right = GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[0, 1], hspace=time_hspace)

    ax11 = fig.add_subplot(inner_left[0])
    ax21 = fig.add_subplot(inner_left[1], sharex=ax11)

    ax12 = fig.add_subplot(inner_right[0])
    ax22 = fig.add_subplot(inner_right[1], sharex=ax12)

    ax31 = fig.add_subplot(outer[1, 0])
    ax32 = fig.add_subplot(outer[1, 1])

    axes = np.array([[ax11, ax12], [ax21, ax22], [ax31, ax32]])

    # ------------------------------------------------------------
    # ROW 1: original time-domain FT + gates
    # ------------------------------------------------------------
    ax11.plot(t1, scaled_abs(signal_t1), color=original_color,
              linewidth=1.3, label="Original FT")
    ax12.plot(t2, scaled_abs(signal_t2), color=original_color,
              linewidth=1.3, label="Original FT")

    for i, gate in enumerate(gates_t1[:2]):
        add_gate_to_axis(ax11, t1, gate, gate_colors[i], f"Gate {i + 1}")

    for i, gate in enumerate(gates_t2[:2]):
        add_gate_to_axis(ax12, t2, gate, gate_colors[i], f"Gate {i + 1}")

    ax11.set_title(device1_label, fontsize=title_fs)
    ax12.set_title(device2_label, fontsize=title_fs)

    ax11.legend(loc="best", fontsize=legend_fs, frameon=False)
    ax12.legend(loc="best", fontsize=legend_fs, frameon=False)

    # ------------------------------------------------------------
    # ROW 2: gated time-domain signals
    # ------------------------------------------------------------
    for i, signal in enumerate(gated_t1[:2]):
        if signal is None or len(signal) == 0:
            continue
        ax21.plot(t1, scaled_abs(signal), color=gate_colors[i],
                  linewidth=1.3, label=f"Gated signal {i + 1}")

    for i, signal in enumerate(gated_t2[:2]):
        if signal is None or len(signal) == 0:
            continue
        ax22.plot(t2, scaled_abs(signal), color=gate_colors[i],
                  linewidth=1.3, label=f"Gated signal {i + 1}")

    if len(gated_t1) > 0:
        ax21.legend(loc="best", fontsize=legend_fs, frameon=False)
    if len(gated_t2) > 0:
        ax22.legend(loc="best", fontsize=legend_fs, frameon=False)

    # Shared time axis: label only on the bottom panel of the block
    ax21.set_xlabel(time_label, fontsize=label_fs)
    ax22.set_xlabel(time_label, fontsize=label_fs)

    # ------------------------------------------------------------
    # ROW 3: raw data (grey, background) + reconstructed spectra
    # ------------------------------------------------------------
    has_raw1 = plot_raw(ax31, freq_raw1, raw_f1, "Device 1")
    has_raw2 = plot_raw(ax32, freq_raw2, raw_f2, "Device 2")

    for i, spectrum in enumerate(gated_f1[:2]):
        if spectrum is None or len(spectrum) == 0:
            continue
        if freq1 is None:
            raise ValueError("freq1 must be provided when gated_f1 is used.")
        if len(freq1) != len(spectrum):
            raise ValueError(
                f"Device 1: freq1 has length {len(freq1)}, "
                f"but gated_f1[{i}] has length {len(spectrum)}."
            )
        ax31.plot(freq1, scaled_abs(spectrum), color=gate_colors[i],
                  linewidth=1.3, zorder=3, label=f"Gated signal {i + 1}")

    for i, spectrum in enumerate(gated_f2[:2]):
        if spectrum is None or len(spectrum) == 0:
            continue
        if freq2 is None:
            raise ValueError("freq2 must be provided when gated_f2 is used.")
        if len(freq2) != len(spectrum):
            raise ValueError(
                f"Device 2: freq2 has length {len(freq2)}, "
                f"but gated_f2[{i}] has length {len(spectrum)}."
            )
        ax32.plot(freq2, scaled_abs(spectrum), color=gate_colors[i],
                  linewidth=1.3, zorder=3, label=f"Gated signal {i + 1}")

    ax31.set_xlabel("Frequency (GHz)", fontsize=label_fs)
    ax32.set_xlabel("Frequency (GHz)", fontsize=label_fs)

    if has_raw1 or len(gated_f1) > 0:
        ax31.legend(loc="best", fontsize=legend_fs, frameon=False)
    if has_raw2 or len(gated_f2) > 0:
        ax32.legend(loc="best", fontsize=legend_fs, frameon=False)

    # ------------------------------------------------------------
    # Y labels: left column only
    # ------------------------------------------------------------
    ax11.set_ylabel(ylabel_row1, fontsize=label_fs)
    ax21.set_ylabel(ylabel_row2, fontsize=label_fs)
    ax31.set_ylabel(ylabel_row3, fontsize=label_fs)

    # ------------------------------------------------------------
    # Axis limits
    # ------------------------------------------------------------
    if time_xlim is not None:
        ax11.set_xlim(time_xlim)   # ax21 follows (shared)
        ax12.set_xlim(time_xlim)   # ax22 follows (shared)

    if freq_xlim is not None:
        ax31.set_xlim(freq_xlim)
        ax32.set_xlim(freq_xlim)

    if time_ylim1 is not None:
        ax11.set_ylim(time_ylim1)
        ax21.set_ylim(time_ylim1)

    if time_ylim2 is not None:
        ax12.set_ylim(time_ylim2)
        ax22.set_ylim(time_ylim2)

    if freq_ylim1 is not None:
        ax31.set_ylim(freq_ylim1)

    if freq_ylim2 is not None:
        ax32.set_ylim(freq_ylim2)

    # ------------------------------------------------------------
    # Axis formatting
    # ------------------------------------------------------------
    for ax in axes.flat:
        setup_axis(ax)
        format_scientific_axis(ax)

    # Hide x tick labels on row 1 (row 2 carries the shared axis)
    ax11.tick_params(labelbottom=False)
    ax12.tick_params(labelbottom=False)

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    return fig, axes


def plot_tof_comparison_2row(
    # Device 1
    t1,
    signal_t1,
    gated_t1=None,
    gates_t1=None,
    freq1=None,
    gated_f1=None,
    freq_raw1=None,      # original frequency axis (Device 1)
    raw_f1=None,         # original S trace (Device 1)

    # Device 2
    t2=None,
    signal_t2=None,
    gated_t2=None,
    gates_t2=None,
    freq2=None,
    gated_f2=None,
    freq_raw2=None,      # original frequency axis (Device 2)
    raw_f2=None,         # original S trace (Device 2)

    # X limits
    time_xlim=None,
    freq_xlim=None,

    # Independent Y limits
    time_ylim1=None,
    time_ylim2=None,
    freq_ylim1=None,
    freq_ylim2=None,

    # Y-axis scaling
    y_scale=1,

    # Labels
    device1_label="Device 1",
    device2_label="Device 2",
    time_label="Time (ns)",

    # Styling
    original_lw=3.5,     # thickness of the original FT (background)
    gated_lw=1.3,        # thickness of the gated traces (on top)
    raw_lw=3.5,          # thickness of the raw frequency data

    # Font sizes
    label_fs=10,         # axis labels (x and y)
    tick_fs=9,           # tick numbers (and the x10^n offset text)
    title_fs=11,         # device titles
    legend_fs=8,         # legends

    # Layout
    row_hspace=0.28,     # gap between row 1 and row 2

    # Saving
    save_path=None,

    # Figure
    figsize=(7.2, 5.2),
):
    """
    Publication-ready comparison plot for time-of-flight spectroscopy.

    Layout (columns: Device 1 | Device 2):
        Row 1: Original time-domain FT (thick, background) with the gate
               regions and the gated signals plotted on top of it
        Row 2: Raw frequency-domain data (thick, background) with the
               gated, reconstructed spectra on top

    y_scale : float
        Plotted quantity is |signal| / y_scale.
    time_ylim1/2, freq_ylim1/2 : tuple or None
        Independent y limits per device (row 1 and row 2 respectively).
    gated_t*, gates_t*, gated_f* : lists with 0, 1 or 2 arrays.
    freq_raw*, raw_f* : original frequency axis and original S trace
        (optional). Plotted behind the gated spectra in row 2.
    label_fs, tick_fs, title_fs, legend_fs : float
        Font sizes (pt) for axis labels, tick numbers, device titles
        and legends.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import ScalarFormatter

    # ------------------------------------------------------------
    # Defaults
    # ------------------------------------------------------------
    gated_t1 = [] if gated_t1 is None else gated_t1
    gated_t2 = [] if gated_t2 is None else gated_t2
    gates_t1 = [] if gates_t1 is None else gates_t1
    gates_t2 = [] if gates_t2 is None else gates_t2
    gated_f1 = [] if gated_f1 is None else gated_f1
    gated_f2 = [] if gated_f2 is None else gated_f2

    if t2 is None or signal_t2 is None:
        raise ValueError("t2 and signal_t2 must be provided for Device 2.")

    # ------------------------------------------------------------
    # Colors
    # ------------------------------------------------------------
    # Lighter slate for the thick background FT so the dark-blue gated
    # trace on top keeps its contrast
    original_color = "#8FA3B8"
    gate_colors = ["#0072B2", "#D55E00"]
    raw_color = "#D9A044"

    # ------------------------------------------------------------
    # Scaling and labels
    # ------------------------------------------------------------
    if y_scale <= 0:
        raise ValueError("y_scale must be positive.")

    if not np.isclose(y_scale, 1):
        exponent = int(np.round(-np.log10(y_scale)))
        scale_suffix = rf"(\times 10^{{{-exponent}}} U)"
    else:
        scale_suffix = ""

    def make_ylabel(core):
        return rf"$|{core}|{scale_suffix}$"

    ylabel_row1 = make_ylabel(r"\mathcal{FT}\;[S_{21}]")
    ylabel_row2 = make_ylabel(r"S_{21}")

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    def scaled_abs(signal):
        return np.abs(signal) / y_scale

    def setup_axis(ax):
        ax.tick_params(direction="in", which="both",
                       top=True, right=True, labelsize=tick_fs)
        ax.minorticks_on()
        ax.tick_params(which="minor", length=3)
        ax.tick_params(which="major", length=5)

    def format_scientific_axis(ax):
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 2))
        ax.yaxis.set_major_formatter(formatter)
        # the "x10^n" offset text at the top of the y axis
        ax.yaxis.get_offset_text().set_fontsize(tick_fs)

    def add_gate_region(ax, t, gate, color):
        """Shaded gate region and dashed boundaries (no legend entry)."""
        gate = np.asarray(gate)
        if len(gate) == 0:
            return
        active = np.where(gate > 1e-6)[0]
        if len(active) == 0:
            return

        t_start = t[active[0]]
        t_stop = t[active[-1]]

        ax.axvspan(t_start, t_stop, color=color, alpha=0.12, lw=0, zorder=0)
        for edge in (t_start, t_stop):
            ax.axvline(edge, color=color, linestyle="--",
                       linewidth=0.9, alpha=0.8, zorder=2)

    def plot_original_ft(ax, t, signal, gates, gated):
        """Thick background FT, gate regions, and gated traces on top."""
        ax.plot(t, scaled_abs(signal), color=original_color,
                linewidth=original_lw, zorder=1, label="Original FT")

        for i, gate in enumerate(gates[:2]):
            add_gate_region(ax, t, gate, gate_colors[i])

        for i, sig in enumerate(gated[:2]):
            if sig is None or len(sig) == 0:
                continue
            ax.plot(t, scaled_abs(sig), color=gate_colors[i],
                    linewidth=gated_lw, zorder=3,
                    label=f"Gated signal {i + 1}")

    def plot_raw(ax, f_raw, s_raw, device_name):
        if s_raw is None or len(s_raw) == 0:
            return False
        if f_raw is None:
            raise ValueError(
                f"{device_name}: the raw frequency axis must be provided "
                f"when the raw trace is used."
            )
        if len(f_raw) != len(s_raw):
            raise ValueError(
                f"{device_name}: raw frequency axis has length {len(f_raw)}, "
                f"but raw trace has length {len(s_raw)}."
            )
        ax.plot(f_raw, scaled_abs(s_raw), color=raw_color,
                linewidth=raw_lw, zorder=1, label="Raw data")
        return True

    def plot_gated_spectra(ax, freq, spectra, device_name):
        for i, spectrum in enumerate(spectra[:2]):
            if spectrum is None or len(spectrum) == 0:
                continue
            if freq is None:
                raise ValueError(
                    f"{device_name}: the frequency axis must be provided "
                    f"when gated spectra are used."
                )
            if len(freq) != len(spectrum):
                raise ValueError(
                    f"{device_name}: frequency axis has length {len(freq)}, "
                    f"but gated spectrum {i} has length {len(spectrum)}."
                )
            ax.plot(freq, scaled_abs(spectrum), color=gate_colors[i],
                    linewidth=gated_lw, zorder=3,
                    label=f"Gated signal {i + 1}")

    # ------------------------------------------------------------
    # Figure and grid
    # ------------------------------------------------------------
    fig = plt.figure(figsize=figsize)

    grid = GridSpec(
        2, 2, figure=fig,
        height_ratios=[1.3, 1.0],
        left=0.11, right=0.98, bottom=0.09, top=0.94,
        wspace=0.22, hspace=row_hspace,
    )

    ax11 = fig.add_subplot(grid[0, 0])
    ax12 = fig.add_subplot(grid[0, 1])
    ax21 = fig.add_subplot(grid[1, 0])
    ax22 = fig.add_subplot(grid[1, 1])

    axes = np.array([[ax11, ax12], [ax21, ax22]])

    # ------------------------------------------------------------
    # ROW 1: original FT + gates + gated signals
    # ------------------------------------------------------------
    plot_original_ft(ax11, t1, signal_t1, gates_t1, gated_t1)
    plot_original_ft(ax12, t2, signal_t2, gates_t2, gated_t2)

    ax11.set_title(device1_label, fontsize=title_fs)
    ax12.set_title(device2_label, fontsize=title_fs)

    ax11.set_xlabel(time_label, fontsize=label_fs)
    ax12.set_xlabel(time_label, fontsize=label_fs)

    ax11.legend(loc="best", fontsize=legend_fs, frameon=False)
    ax12.legend(loc="best", fontsize=legend_fs, frameon=False)

    # ------------------------------------------------------------
    # ROW 2: raw data + reconstructed spectra
    # ------------------------------------------------------------
    has_raw1 = plot_raw(ax21, freq_raw1, raw_f1, "Device 1")
    has_raw2 = plot_raw(ax22, freq_raw2, raw_f2, "Device 2")

    plot_gated_spectra(ax21, freq1, gated_f1, "Device 1")
    plot_gated_spectra(ax22, freq2, gated_f2, "Device 2")

    ax21.set_xlabel("Frequency (GHz)", fontsize=label_fs)
    ax22.set_xlabel("Frequency (GHz)", fontsize=label_fs)

    if has_raw1 or len(gated_f1) > 0:
        ax21.legend(loc="best", fontsize=legend_fs, frameon=False)
    if has_raw2 or len(gated_f2) > 0:
        ax22.legend(loc="best", fontsize=legend_fs, frameon=False)

    # ------------------------------------------------------------
    # Y labels: left column only
    # ------------------------------------------------------------
    ax11.set_ylabel(ylabel_row1, fontsize=label_fs)
    ax21.set_ylabel(ylabel_row2, fontsize=label_fs)

    # ------------------------------------------------------------
    # Axis limits
    # ------------------------------------------------------------
    if time_xlim is not None:
        ax11.set_xlim(time_xlim)
        ax12.set_xlim(time_xlim)

    if freq_xlim is not None:
        ax21.set_xlim(freq_xlim)
        ax22.set_xlim(freq_xlim)

    if time_ylim1 is not None:
        ax11.set_ylim(time_ylim1)

    if time_ylim2 is not None:
        ax12.set_ylim(time_ylim2)

    if freq_ylim1 is not None:
        ax21.set_ylim(freq_ylim1)

    if freq_ylim2 is not None:
        ax22.set_ylim(freq_ylim2)

    # ------------------------------------------------------------
    # Axis formatting
    # ------------------------------------------------------------
    for ax in axes.flat:
        setup_axis(ax)
        format_scientific_axis(ax)

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    return fig, axes

class Analysis_timeD(Analysis_PSWS):
    def __init__(self, address , file_name:str, sample:str , setup:str, geo:str, device: str, **kwargs):
        super().__init__(address , file_name, sample, setup, geo, device, **kwargs)
        self.freq = np.asarray(self.freqs) #Should be already in GHz
        self.time_axis = np.asarray([]) #in ns

    def spectrum_to_time(
            self,
            dS21R,
            dS21I,
            f_min=0.1,
            f_max=4.7,
            window=None,
            zero_padding=1,
            plot=True,
    ):
        """
        Convert a complex frequency-domain spectrum into the time domain.

        The measured positive-frequency spectrum is extended to negative
        frequencies using Hermitian symmetry. Frequencies between zero
        and f_min are set to zero.

        Parameters
        ----------
        dS21R, dS21I : arrays
            Real and imaginary parts of S21.
        f_min, f_max : float
            Frequency range used for the IFFT, in GHz.
        window : str or None
            "hann", "hamming", "blackman", or None.
        zero_padding : int
            Zero-padding factor.
        plot : bool
            Plot the time-domain signal.

        Sets
        -------
        self.time_axis : array
            Time axis in ns.

        Returns
        -------
        S21_t : array
            Complex time-domain signal.
        """
        S21 = np.asarray(dS21R) + 1j * np.asarray(dS21I)

        # ---------------------------------------------------------
        # Select frequency range
        # ---------------------------------------------------------

        mask = (self.freq >= f_min) & (self.freq <= f_max)

        f = self.freq[mask]
        S = S21[mask]

        # Check frequency spacing
        df_array = np.diff(f)
        df = np.mean(df_array)

        if not np.allclose(df_array, df, rtol=1e-4, atol=1e-12):
            raise ValueError(
                "Frequency points must be uniformly spaced."
            )

        # ---------------------------------------------------------
        # Frequency window
        # ---------------------------------------------------------

        if window is None:

            win = np.ones(len(S))

        elif window.lower() == "hann":

            win = np.hanning(len(S))

        elif window.lower() == "hamming":

            win = np.hamming(len(S))

        elif window.lower() == "blackman":

            win = np.blackman(len(S))

        else:

            raise ValueError(
                "window must be 'hann', 'hamming', 'blackman', or None"
            )

        S_windowed = S * win

        # ---------------------------------------------------------
        # Construct full Hermitian spectrum
        # ---------------------------------------------------------

        # Number of measured positive-frequency points
        n_positive = len(S_windowed)

        # Number of zero-frequency bins between 0 and f_min
        n_gap = int(np.round(f[0] / df)) - 1

        if n_gap < 0:
            n_gap = 0

        # Zero-frequency region
        S_gap = np.zeros(n_gap, dtype=complex)

        # Negative-frequency spectrum
        S_negative = np.conj(S_windowed[::-1])

        # Full spectrum:
        #
        # negative frequencies
        #       ↓
        # S(-fmax) ... S(-fmin)
        #
        # zero-frequency gap
        #
        # S(+fmin) ... S(+fmax)
        #
        S_full = np.concatenate([
            S_negative,
            S_gap,
            [0],
            S_gap,
            S_windowed
        ])

        # ---------------------------------------------------------
        # Zero padding
        # ---------------------------------------------------------

        n_original = len(S_full)
        n_fft = zero_padding * n_original

        # ---------------------------------------------------------
        # IFFT
        # ---------------------------------------------------------

        S21_t = np.fft.ifft(
            np.fft.ifftshift(S_full),
            n=n_fft
        )

        # ---------------------------------------------------------
        # Time axis
        # ---------------------------------------------------------

        # df in GHz -> dt in ns
        dt = 1 / (n_fft * df)

        self.time_axis = np.arange(n_fft) * dt

        # ---------------------------------------------------------
        # Plot
        # ---------------------------------------------------------

        if plot:
            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=self.time_axis,
                    y=np.abs(S21_t),
                    mode="lines",
                    name="|S₂₁(t)|"
                )
            )

            fig.update_layout(
                xaxis_title="Time (ns)",
                yaxis_title="|S₂₁(t)|",
                template="plotly_white",
                width=900,
                height=500,
                hovermode="x unified",
            )

            fig.show()
        return S21_t

    def time_gate(
            self,
            signal,
            t_start,
            t_stop,
            gate_type="heaviside",
            tukey_alpha=0.5,
            plot=True,
    ):
        """
        Apply a time-domain gate to a complex signal.

        Parameters
        ----------
        signal : array
            Complex time-domain signal.
        t_start, t_stop : float
            Start and stop times of the gate in ns.
        gate_type : str
            "heaviside" or "tukey".
        tukey_alpha : float
            Tukey parameter between 0 and 1.
            Only used when gate_type="tukey".
            0 = rectangular, 1 = Hann.
        plot : bool
            Plot original signal, gated signal, and gate.

        Returns
        -------
        gated_signal : array
            Complex gated signal.
        gate : array
            Gate function.
        """

        signal = np.asarray(signal)

        if t_stop <= t_start:
            raise ValueError("t_stop must be larger than t_start.")

        # ---------------------------------------------------------
        # HEAVISIDE GATE
        # ---------------------------------------------------------

        if gate_type.lower() == "heaviside":

            gate = (
                    np.heaviside(self.time_axis - t_start, 1)
                    - np.heaviside(self.time_axis - t_stop, 1)
            )

        # ---------------------------------------------------------
        # TUKEY GATE
        # ---------------------------------------------------------

        elif gate_type.lower() == "tukey":

            if not 0 <= tukey_alpha <= 1:
                raise ValueError("tukey_alpha must be between 0 and 1.")

            gate = np.zeros_like(self.time_axis, dtype=float)

            inside = (self.time_axis >= t_start) & (self.time_axis <= t_stop)

            x = (self.time_axis[inside] - t_start) / (t_stop - t_start)

            gate_inside = np.ones_like(x)

            # alpha = 0 → rectangular gate
            if tukey_alpha > 0:
                # Rising edge
                rising = x < tukey_alpha / 2

                gate_inside[rising] = 0.5 * (
                        1
                        + np.cos(
                    2 * np.pi / tukey_alpha
                    * (x[rising] - tukey_alpha / 2)
                )
                )

                # Falling edge
                falling = x > 1 - tukey_alpha / 2

                gate_inside[falling] = 0.5 * (
                        1
                        + np.cos(
                    2 * np.pi / tukey_alpha
                    * (x[falling] - 1 + tukey_alpha / 2)
                )
                )

            gate[inside] = gate_inside

        else:

            raise ValueError(
                "gate_type must be 'heaviside' or 'tukey'."
            )

        # ---------------------------------------------------------
        # APPLY GATE
        # ---------------------------------------------------------

        gated_signal = signal * gate

        # ============================================================
        # 3. Plot
        # ============================================================

        if plot:
            # --------------------------------------------------------
            # Time domain
            # --------------------------------------------------------

            fig_time = go.Figure()

            fig_time.add_trace(
                go.Scatter(
                    x=self.time_axis,
                    y=signal.real,
                    mode="lines",
                    name="Original"
                )
            )

            fig_time.add_trace(
                go.Scatter(
                    x=self.time_axis,
                    y=np.abs(signal),
                    mode="lines",
                    name="|Original|"
                )
            )

            fig_time.add_trace(
                go.Scatter(
                    x=self.time_axis,
                    y=gated_signal.real,
                    mode="lines",
                    name="Gated"
                )
            )

            # Show gate on secondary scale
            gate_scaled = gate * np.max(np.abs(signal))

            fig_time.add_trace(
                go.Scatter(
                    x=self.time_axis,
                    y=gate_scaled,
                    mode="lines",
                    name="Gate",
                    line=dict(dash="dash")
                )
            )

            fig_time.update_layout(
                title=f"Time gate: {t_start}–{t_stop} ns",
                xaxis_title="Time (ns)",
                yaxis_title="Signal(t)",
                template="plotly_white",
                width=900,
                height=500,
                hovermode="x unified",
                xaxis_range=[0, 200],
                # yaxis_range=[-2e-5, 2e-5],
            )

            fig_time.show()

            # --------------------------------------------------------
            # Frequency domain
            # --------------------------------------------------------

            freq, spectrum = self.time_to_spectrum(
                gated_signal,
                plot=True
            )

        return gated_signal, gate

    def time_to_spectrum(
            self,
            signal,
            zero_padding=1,
            positive_only=True,
            plot=False,
    ):
        """
        Convert a complex time-domain signal into the frequency domain.

        Parameters
        ----------
        signal : array
            Complex time-domain signal.
        zero_padding : int
            Zero-padding factor.
        positive_only : bool
            If True, return only positive frequencies.
        plot : bool
            Plot the spectrum.

        Returns
        -------
        freq : array
            Frequency in GHz.
        spectrum : array
            Complex frequency-domain spectrum.
        """
        signal = np.asarray(signal)

        # Check uniform time spacing
        dt_array = np.diff(self.time_axis)
        dt = np.mean(dt_array)

        if not np.allclose(dt_array, dt, rtol=1e-4, atol=1e-12):
            raise ValueError("Time points must be uniformly spaced.")

        n = len(signal)
        n_fft = zero_padding * n

        # FFT
        spectrum = np.fft.fft(signal, n=n_fft)
        freq = np.fft.fftfreq(n_fft, d=dt)

        # Sort frequencies from negative to positive
        spectrum = np.fft.fftshift(spectrum)
        freq = np.fft.fftshift(freq)

        # Keep only positive frequencies
        if positive_only:
            mask = freq >= 0
            freq = freq[mask]
            spectrum = spectrum[mask]

        if plot:
            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=freq,
                    y=np.abs(spectrum),
                    mode="lines",
                    name="Spectrum"
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=freq,
                    y=spectrum.real,
                    mode="lines",
                    name="Re(Spectrum)"
                )
            )

            fig.update_layout(
                xaxis_title="Frequency (GHz)",
                yaxis_title="S",
                template="plotly_white",
                width=900,
                height=500,
                hovermode="x unified",
                xaxis_range=[3.5, 4.6],
            )

            fig.show()

        return freq, spectrum

    def plot_dataVSgated(self, dS_data, freq_gated, dS_gated):
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=self.freq,
                y=dS_data.real,
                mode="lines",
                name="Re(dS_data)"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=self.freq,
                y=np.abs(dS_data),
                mode="lines",
                name="|dS_data|"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=freq_gated,
                y=dS_gated.real,
                mode="lines",
                name="Re(dS_gated)"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=freq_gated,
                y=np.abs(dS_gated),
                mode="lines",
                name="|dS_gated|"
            )
        )

        fig.update_layout(
            title=f"Raw vs gated",
            xaxis_title="frequency (GHz)",
            yaxis_title="dS",
            template="plotly_white",
            width=900,
            height=500,
            hovermode="x unified",
            xaxis_range=[3.5, 4.6],
        )

        fig.show()