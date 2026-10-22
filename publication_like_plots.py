import numpy as np
import matplotlib.pyplot as plt

def plot_complex_fit(
        x_exp,
        y_exp,
        x_fit,
        y_fit,
        x_fit_2=None,
        y_fit_2=None,
        x_fit_3=None,
        y_fit_3=None,

        ij="21",  # i,j=1,2

        # --------------------------------------------------
        # What to plot
        # --------------------------------------------------
        plot_imag=True,
        plot_fit_2=True,
        plot_fit_3=True,

        # --------------------------------------------------
        # Axis limits
        # --------------------------------------------------
        xlim=None,
        ylim=None,

        # --------------------------------------------------
        # Labels
        # --------------------------------------------------
        xlabel="Frequency (GHz)",
        ylabel="Response (a.u.)",
        xlabel_fontsize=13,
        ylabel_fontsize=13,

        # --------------------------------------------------
        # Tick formatting
        # --------------------------------------------------
        tick_fontsize=11,
        tick_width=1.2,
        tick_length=5,

        # --------------------------------------------------
        # Axis lines
        # --------------------------------------------------
        spine_width=1.2,

        # --------------------------------------------------
        # Experimental data
        # --------------------------------------------------
        markersize=4,

        # --------------------------------------------------
        # Fits
        # --------------------------------------------------
        linewidth=2,

        # --------------------------------------------------
        # Colors
        # --------------------------------------------------

        # Experimental
        color_re_exp= None,  # "#1F4E79",
        color_im_exp= None,  # "#5B2C83",
        color_magnitude= None,

        # Fits -- brighter
        color_re_fit="#4EA3FF",
        color_re_fit_2= None,
        color_re_fit_3= None,
        color_im_fit= None,
        color_mag_fit= None,

        # --------------------------------------------------
        # Magnitude background
        # --------------------------------------------------
        magnitude_linewidth=3,

        # --------------------------------------------------
        # Legend
        # --------------------------------------------------
        legend_fontsize=10,
        legend_frame=False,
        legend_loc="best",

        # --------------------------------------------------
        # General
        # --------------------------------------------------
        font="Arial",
        figsize=(5, 4),

        # --------------------------------------------------
        # Save
        # --------------------------------------------------
        save=False,
        filename="complex_fit.png",
        dpi=600
):
    # Experimental
    # Default colors depending on ij
    if ij == "21":
        if color_re_exp is None:
            color_re_exp = "#3978A8"
        if color_im_exp is None:
            color_im_exp = "#C65A5A"
        if color_magnitude is None:
            color_magnitude = "#D9A044"

    elif ij == "12":
        if color_re_exp is None:
            color_re_exp = "#00AFC0"
        if color_im_exp is None:
            color_im_exp = "#00CC96"
        if color_magnitude is None:
            color_magnitude = "#8B70B0"

    # ==================================================
    # Figure
    # ==================================================

    fig, ax = plt.subplots(figsize=figsize)

    # ==================================================
    # Experimental magnitude
    # ==================================================

    ax.plot(
        x_exp,
        np.abs(y_exp),
        "o",
        color=color_magnitude,
        markersize=markersize,
        # markevery=20,
        markerfacecolor='none',
        label=rf"Mag($\Delta \hat{{S}}_{{{ij}}}$)",
        zorder=1
    )

    # # ==================================================
    # # Experimental real
    # # ==================================================

    # ax.plot(
    #     x_exp,
    #     np.real(y_exp),
    #     "o",
    #     color=color_re_exp,
    #     markersize=markersize,
    #     markevery=5,
    #     markerfacecolor='none',
    #     label=rf"Re($\Delta \hat{{S}}_{{{ij}}}$)",
    #     zorder=2
    # )

    # ==================================================
    # Experimental imaginary
    # ==================================================

    if plot_imag:
        ax.plot(
            x_exp,
            np.imag(y_exp),
            "o",
            color=color_im_exp,
            markersize=markersize,
            # markevery=5,
            markerfacecolor='none',
            label=rf"Im($\Delta \hat{{S}}_{{{ij}}}$)",
            zorder=2
        )

    # # ==================================================
    # # Fit 1 — real
    # # ==================================================

    # ax.plot(
    #     x_fit,
    #     np.real(y_fit),
    #     "--",
    #     color=color_re_fit,
    #     linewidth=linewidth,
    #     label="Re(Fit)",
    #     zorder=3
    # )

    # ==================================================
    # Fit 1 — imaginary
    # ==================================================

    if plot_imag:
        ax.plot(
            x_fit,
            np.imag(y_fit),
            "-",
            color=color_im_fit,
            linewidth=linewidth,
            label="Im(Fit)",
            zorder=4
        )

    # ==================================================
    # Fit 1 — mag
    # ==================================================

    if plot_imag:
        ax.plot(
            x_fit,
            np.abs(y_fit),
            "-",
            color=color_mag_fit,
            linewidth=linewidth,
            label="Mag(Fit)",
            zorder=5
        )

    # ==================================================
    # Fit 2
    # ==================================================

    if plot_fit_2 and x_fit_2 is not None and y_fit_2 is not None:

        # # Real
        # ax.plot(
        #     x_fit_2,
        #     np.real(y_fit_2),
        #     "-",
        #     color=color_re_fit_2,
        #     linewidth=linewidth,
        #     label="Re(wave1)",
        #     zorder=6
        # )

        # Imaginary
        if plot_imag:
            ax.plot(
                x_fit_2,
                np.imag(y_fit_2),
                "--",
                color=color_re_fit_2,
                linewidth=linewidth,
                label="Im(wave1)",
                zorder=6
            )

    # ==================================================
    # Fit 3
    # ==================================================

    if plot_fit_3 and x_fit_3 is not None and y_fit_3 is not None:

        # # Real
        # ax.plot(
        #     x_fit_3,
        #     np.real(y_fit_3),
        #     "-",
        #     color=color_re_fit_3,
        #     linewidth=linewidth,
        #     label="Re(wave2)",
        #     zorder=7
        # )

        # Imaginary
        if plot_imag:
            ax.plot(
                x_fit_3,
                np.imag(y_fit_3),
                "--",
                color=color_re_fit_3,
                linewidth=linewidth,
                label="Im(wave2)",
                zorder=7
            )

    # ==================================================
    # Axis limits
    # ==================================================

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    # ==================================================
    # Labels
    # ==================================================

    ax.set_xlabel(
        xlabel,
        fontsize=xlabel_fontsize,
        fontname=font
    )

    ax.set_ylabel(
        ylabel,
        fontsize=ylabel_fontsize,
        fontname=font
    )

    # ==================================================
    # Ticks
    # ==================================================

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=tick_fontsize,
        width=tick_width,
        length=tick_length
    )

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname(font)

    # ==================================================
    # Spines
    # ==================================================

    for spine in ax.spines.values():
        spine.set_linewidth(spine_width)

    # ==================================================
    # Legends
    # ==================================================

    # Get all plotted handles and labels
    handles, labels = ax.get_legend_handles_labels()

    # First legend: everything except wave 1 and wave 2
    handles_main = []
    labels_main = []

    # Second legend: wave 1 and wave 2
    handles_waves = []
    labels_waves = []

    for handle, label in zip(handles, labels):

        if label in ["Re(wave 1)", "Re(wave 2)", "Im(wave1)", "Im(wave2)"]:
            handles_waves.append(handle)
            labels_waves.append(label)

        else:
            handles_main.append(handle)
            labels_main.append(label)

    # Upper-right legend
    legend_main = ax.legend(
        handles_main,
        labels_main,
        fontsize=legend_fontsize,
        frameon=legend_frame,
        loc="upper right"
    )

    # Lower-right legend
    if handles_waves:
        legend_waves = ax.legend(
            handles_waves,
            labels_waves,
            fontsize=legend_fontsize,
            frameon=legend_frame,
            loc="lower right"
        )

        # Keep the first legend
        ax.add_artist(legend_main)

    # ---------------------------------------------------------
    # Zero reference line
    # ---------------------------------------------------------

    ax.axhline(
        0,
        color='black',
        linewidth=0.8,
        linestyle='-',
        alpha=0.5,
        zorder=2
    )

    # ==================================================
    # Final
    # ==================================================

    fig.tight_layout()

    if save:
        fig.savefig(
            filename,
            dpi=dpi,
            bbox_inches="tight"
        )
    plt.minorticks_on()
    plt.show()

    return fig, ax