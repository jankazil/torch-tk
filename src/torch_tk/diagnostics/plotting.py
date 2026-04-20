'''
Plotting utilities for sample-resolved diagnostics.

This module provides a function to estimate and plot probability density functions
of the per-sample loss across one or more diagnostics objects and training epochs.
'''

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from .diagnostics import Diagnostics


def plot_positive_loss_kde_pdf(
    diagnostics: Diagnostics | list[Diagnostics],
    plot_file: Path | str | None = None,
    title: str | None = None,
    loss_name='Loss',
    font_factor=1.5,
    figsize=(9, 6),
    xlim=None,
    ylim=None,
    xlog=False,
    ylog=False,
    per_dlog10: bool = False,
    bin_n=100,
    show_plot=True,
    verbose=True,
    epoch_skip=1,
) -> None:
    '''
    Plot kernel density estimates of the per-sample loss probability density function (PDF).

    The loss values must be > 0 as the kernel density estimation is done in log-space of
    the loss values.

    This function accepts one Diagnostics object or a list of them. For each selected
    epoch in each diagnostic, it takes the per-sample losses for that epoch, determines the
    probability density function using a Gaussian kernel density estimate in log-loss space,
    and plots the resulting PDFs on a shared axis.

    The x-axis represents loss by default. The density can optionally be converted
    from dP/d(loss) to dP/dlog10(loss), which is useful for comparing the probability
    density across scales on a logarithmic x-axis.

    Arguments:

        diagnostics:
            A Diagnostics instance or a list of Diagnostics instances. Each diagnostic
            must provide per-sample loss data indexed by epoch, and all loss values
            must be strictly positive.
        plot_file:
            Output path for saving the figure. If None, the figure is not written to disk.
        title:
            Figure title.
        loss_name:
            Label used for the x-axis quantity. Default is 'Loss'.
        font_factor:
            Multiplicative scaling applied to all font sizes in the figure.
        figsize:
            Figure size passed to matplotlib.
        xlim:
            Optional x-axis limits.
        ylim:
            Optional y-axis limits.
        xlog:
            If True, use a logarithmic x-axis unless overridden by per_dlog10 handling.
        ylog:
            If True, use a logarithmic y-axis.
        per_dlog10:
            If True, plot density per unit log10 of the x quantity rather than per unit
            x. This also forces logarithmic scaling on the x-axis.
        bin_n:
            Number of grid points used to evaluate each KDE.
        show_plot:
            If True, display the figure with plt.show(). Otherwise close it after optional
            saving.
        verbose:
            If True, print the saved plot path when plot_file is provided.
        epoch_skip:
            Plot every epoch_skip-th stored epoch.

    Notes:

        This function assumes that diagnostic.per_sample_loss is indexed in the same order
        as diagnostic.epoch, with one per-sample loss array per stored epoch.
    '''

    # Validate input

    if epoch_skip < 1:
        raise ValueError('epoch_skip must be >= 1')

    diagnostics = diagnostics if isinstance(diagnostics, list) else [diagnostics]

    for diagnostic in diagnostics:
        if diagnostic.per_sample_loss is None:
            raise ValueError('Each diagnostic must contain per-sample loss data.')

    loss_min = min([diagnostic.per_sample_loss.min().item() for diagnostic in diagnostics])
    loss_max = max([diagnostic.per_sample_loss.max().item() for diagnostic in diagnostics])

    if loss_min <= 0:
        raise ValueError('Each diagnostic must contain only positive loss values.')

    # Do your worst

    fig, ax = plt.subplots(figsize=figsize, nrows=1, ncols=1, squeeze=False)

    plt.suptitle(title, y=0.94, fontsize=8)

    # Construct PDF grid

    grid_min = 0.1 * loss_min
    grid_max = 10 * loss_max

    grid = np.logspace(np.log10(grid_min), np.log10(grid_max), bin_n)

    pdfs = []

    # Calculate and plot PDFs

    for diagnostic in diagnostics:
        for epoch_i in range(len(diagnostic.epoch) - 1, -1, -epoch_skip):
            epoch = diagnostic.epoch[epoch_i]

            # Sample-resolved loss
            model_losses = diagnostic.per_sample_loss[epoch_i].to(device='cpu')

            # Construct the kernel density estimate
            kde = gaussian_kde(model_losses.log())

            # Evaluate the probability density on the grid
            pdf = kde(np.log(grid)) / grid

            # Convert to probability density per unit log10(L): dP/dlog10(L) = ln(10) * L * dP/dL
            # This will allow comparing values across scales when plotting L using a logarithmic grid.
            if per_dlog10:
                pdf = np.log(10) * grid * pdf

            pdfs.append(pdf)

            label = (
                diagnostic.model
                + ', '
                + diagnostic.optimizer
                + ', lr = '
                + str(diagnostic.learning_rate)
                + ', batch size = '
                + str(diagnostic.batch_size)
                + ', epoch = '
                + str(epoch)
            )

            if diagnostic.description:
                label = label + ', ' + diagnostic.description

            # Plot
            ax[0, 0].plot(grid, pdf, label=label)

    positive_pdf_mins = [pdf[pdf > 0].min() for pdf in pdfs if np.any(pdf > 0)]
    if positive_pdf_mins:
        pdf_min = min(positive_pdf_mins).item()
    else:
        pdf_min = 1e-12

    pdf_max = max([pdf[pdf > 0].max() for pdf in pdfs]).item()

    ax[0, 0].legend(loc='best', fontsize=7, frameon=False)

    ax[0, 0].set_xlabel(loss_name)

    if per_dlog10:
        ax[0, 0].set_ylabel('Probability density dp/dlog10(' + loss_name + ')')
    else:
        ax[0, 0].set_ylabel('Probability density dp/d(' + loss_name + ')')

    if per_dlog10:
        ax[0, 0].set_xscale('log')
    else:
        if xlog:
            ax[0, 0].set_xscale('log')
        else:
            ax[0, 0].set_xscale('linear')

    if ylog:
        ax[0, 0].set_yscale('log')
    else:
        ax[0, 0].set_yscale('linear')

    if xlim:
        ax[0, 0].set_xlim(xlim)
    else:
        ax[0, 0].set_xlim([grid_min, grid_max])
    if ylim:
        ax[0, 0].set_ylim(ylim)
    else:
        ax[0, 0].set_ylim([pdf_min, pdf_max])

    # Increase all font sizes by a given factor
    for text in fig.findobj(match=lambda artist: hasattr(artist, 'get_fontsize')):
        text.set_fontsize(text.get_fontsize() * font_factor)

    if plot_file:
        plot_file = Path(plot_file)
        plot_dir = plot_file.parent
        os.makedirs(plot_dir, exist_ok=True)
        fig.savefig(plot_file, bbox_inches='tight')

        if verbose:
            print('Created plot', plot_file)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return


def plot_positive_loss_hist_pdf(
    diagnostics: Diagnostics | list[Diagnostics],
    plot_file: Path | str | None = None,
    title: str | None = None,
    loss_name='Loss',
    font_factor=1.5,
    figsize=(9, 6),
    xlim=None,
    ylim=None,
    xlog=False,
    ylog=False,
    per_dlog10: bool = False,
    bin_n=25,
    show_plot=True,
    verbose=True,
    epoch_skip=1,
) -> None:
    '''
    Plot histogram-based estimates of the per-sample loss probability density function (PDF)
    using logarithmically spaced bins.

    The loss values must be > 0.

    This function accepts one Diagnostics object or a list of them. For each selected
    epoch in each diagnostic, it takes the per-sample losses for that epoch, constructs
    a histogram on logarithmically spaced bins, converts that histogram into a probability
    density function, and plots the resulting PDFs on a shared axis.

    The x-axis represents loss by default. The density can optionally be converted
    from dP/d(loss) to dP/dlog10(loss), which is useful for comparing the probability
    density across scales on a logarithmic x-axis.

    Arguments:

        diagnostics:
            A Diagnostics instance or a list of Diagnostics instances. Each diagnostic
            must provide per-sample loss data indexed by epoch, and all loss values
            must be strictly positive.
        plot_file:
            Output path for saving the figure. If None, the figure is not written to disk.
        title:
            Figure title.
        loss_name:
            Label used for the x-axis quantity. Default is 'Loss'.
        font_factor:
            Multiplicative scaling applied to all font sizes in the figure.
        figsize:
            Figure size passed to matplotlib.
        xlim:
            Optional x-axis limits.
        ylim:
            Optional y-axis limits.
        xlog:
            If True, use a logarithmic x-axis unless overridden by per_dlog10 handling.
        ylog:
            If True, use a logarithmic y-axis.
        per_dlog10:
            If True, plot density per unit log10 of the x quantity rather than per unit
            x. This also forces logarithmic scaling on the x-axis.
        bin_n:
            Number of logarithmic bins.
        show_plot:
            If True, display the figure with plt.show(). Otherwise close it after optional
            saving.
        verbose:
            If True, print the saved plot path when plot_file is provided.
        epoch_skip:
            Plot every epoch_skip-th stored epoch.

    Notes:

        This function assumes that diagnostic.per_sample_loss is indexed in the same order
        as diagnostic.epoch, with one per-sample loss array per stored epoch.
    '''

    # Validate input

    if epoch_skip < 1:
        raise ValueError('epoch_skip must be >= 1')

    diagnostics = diagnostics if isinstance(diagnostics, list) else [diagnostics]

    for diagnostic in diagnostics:
        if diagnostic.per_sample_loss is None:
            raise ValueError('Each diagnostic must contain per-sample loss data.')

    loss_min = min([diagnostic.per_sample_loss.min().item() for diagnostic in diagnostics])
    loss_max = max([diagnostic.per_sample_loss.max().item() for diagnostic in diagnostics])

    if loss_min <= 0:
        raise ValueError('Each diagnostic must contain only positive loss values.')

    # Do your worst

    fig, ax = plt.subplots(figsize=figsize, nrows=1, ncols=1, squeeze=False)

    plt.suptitle(title, y=0.94, fontsize=8)

    # Construct logarithmic/geometric bins

    grid_min = 0.1 * loss_min
    grid_max = 10 * loss_max

    bin_edges = np.logspace(np.log10(grid_min), np.log10(grid_max), bin_n + 1)
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])  # geometric centers
    bin_widths = np.diff(bin_edges)

    pdfs = []

    # Calculate and plot PDFs

    for diagnostic in diagnostics:
        for epoch_i in range(len(diagnostic.epoch) - 1, -1, -epoch_skip):
            epoch = diagnostic.epoch[epoch_i]

            # Sample-resolved loss
            model_losses = diagnostic.per_sample_loss[epoch_i].to(device='cpu').numpy()

            # Histogram counts in logarithmic bins
            counts, _ = np.histogram(model_losses, bins=bin_edges, density=False)

            # Probability density per unit loss: dP/dL
            pdf = counts / (counts.sum() * bin_widths)

            # Convert to probability density per unit log10(L):
            # dP/dlog10(L) = ln(10) * L * dP/dL
            if per_dlog10:
                pdf = np.log(10) * bin_centers * pdf

            pdfs.append(pdf)

            label = (
                diagnostic.model
                + ', '
                + diagnostic.optimizer
                + ', lr = '
                + str(diagnostic.learning_rate)
                + ', batch size = '
                + str(diagnostic.batch_size)
                + ', epoch = '
                + str(epoch)
            )

            if diagnostic.description:
                label = label + ', ' + diagnostic.description

            # Plot using bin centers
            ax[0, 0].plot(bin_centers, pdf, label=label)

    positive_pdf_mins = [pdf[pdf > 0].min() for pdf in pdfs if np.any(pdf > 0)]
    if positive_pdf_mins:
        pdf_min = min(positive_pdf_mins).item()
    else:
        pdf_min = 1e-12

    pdf_max = max([pdf.max() for pdf in pdfs]).item()

    ax[0, 0].legend(loc='best', fontsize=7, frameon=False)

    ax[0, 0].set_xlabel(loss_name)

    if per_dlog10:
        ax[0, 0].set_ylabel('Probability density dp/dlog10(' + loss_name + ')')
    else:
        ax[0, 0].set_ylabel('Probability density dp/d(' + loss_name + ')')

    if per_dlog10:
        ax[0, 0].set_xscale('log')
    else:
        if xlog:
            ax[0, 0].set_xscale('log')
        else:
            ax[0, 0].set_xscale('linear')

    if ylog:
        ax[0, 0].set_yscale('log')
    else:
        ax[0, 0].set_yscale('linear')

    if xlim:
        ax[0, 0].set_xlim(xlim)
    else:
        ax[0, 0].set_xlim([grid_min, grid_max])

    if ylim:
        ax[0, 0].set_ylim(ylim)
    else:
        ax[0, 0].set_ylim([pdf_min, pdf_max])

    # Increase all font sizes by a given factor
    for text in fig.findobj(match=lambda artist: hasattr(artist, 'get_fontsize')):
        text.set_fontsize(text.get_fontsize() * font_factor)

    if plot_file:
        plot_file = Path(plot_file)
        plot_dir = plot_file.parent
        os.makedirs(plot_dir, exist_ok=True)
        fig.savefig(plot_file, bbox_inches='tight')

        if verbose:
            print('Created plot', plot_file)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return


def plot_positive_loss_hist_1st_moment_density(
    diagnostics: Diagnostics | list[Diagnostics],
    plot_file: Path | str | None = None,
    title: str | None = None,
    loss_name='Loss',
    font_factor=1.5,
    figsize=(9, 6),
    xlim=None,
    ylim=None,
    xlog=False,
    ylog=False,
    per_dlog10: bool = False,
    bin_n=25,
    show_plot=True,
    verbose=True,
    epoch_skip=1,
) -> None:
    '''
    Plot histogram-based estimates of the first-moment density of the per-sample loss
    distribution using logarithmically spaced bins.

    The loss values must be > 0.

    This function accepts one Diagnostics object or a list of them. For each selected
    epoch in each diagnostic, it takes the per-sample losses for that epoch, constructs
    a histogram on logarithmically spaced bins, converts that histogram into an estimate
    of the loss PDF, and then multiplies by loss to plot the first-moment density.

    By default, the plotted quantity is

        loss * dP/d(loss)

    which represents the contribution to the first moment per unit loss.

    Optionally, this can be converted to the corresponding quantity per unit log10(loss),

        loss * dP/dlog10(loss)

    which is useful for comparing first-moment contributions across scales on a
    logarithmic x-axis.

    Arguments:

        diagnostics:
            A Diagnostics instance or a list of Diagnostics instances. Each diagnostic
            must provide per-sample loss data indexed by epoch, and all loss values
            must be strictly positive.
        plot_file:
            Output path for saving the figure. If None, the figure is not written to disk.
        title:
            Figure title.
        loss_name:
            Label used for the x-axis quantity. Default is 'Loss'.
        font_factor:
            Multiplicative scaling applied to all font sizes in the figure.
        figsize:
            Figure size passed to matplotlib.
        xlim:
            Optional x-axis limits.
        ylim:
            Optional y-axis limits.
        xlog:
            If True, use a logarithmic x-axis unless overridden by per_dlog10 handling.
        ylog:
            If True, use a logarithmic y-axis.
        per_dlog10:
            If True, plot the first-moment density per unit log10 of the x quantity
            rather than per unit x. This also forces logarithmic scaling on the x-axis.
        bin_n:
            Number of logarithmic bins.
        show_plot:
            If True, display the figure with plt.show(). Otherwise close it after optional
            saving.
        verbose:
            If True, print the saved plot path when plot_file is provided.
        epoch_skip:
            Plot every epoch_skip-th stored epoch.

    Notes:

        This function assumes that diagnostic.per_sample_loss is indexed in the same order
        as diagnostic.epoch, with one per-sample loss array per stored epoch.
    '''

    # Validate input

    if epoch_skip < 1:
        raise ValueError('epoch_skip must be >= 1')

    diagnostics = diagnostics if isinstance(diagnostics, list) else [diagnostics]

    for diagnostic in diagnostics:
        if diagnostic.per_sample_loss is None:
            raise ValueError('Each diagnostic must contain per-sample loss data.')

    loss_min = min([diagnostic.per_sample_loss.min().item() for diagnostic in diagnostics])
    loss_max = max([diagnostic.per_sample_loss.max().item() for diagnostic in diagnostics])

    if loss_min <= 0:
        raise ValueError('Each diagnostic must contain only positive loss values.')

    # Do your worst

    fig, ax = plt.subplots(figsize=figsize, nrows=1, ncols=1, squeeze=False)

    plt.suptitle(title, y=0.94, fontsize=8)

    # Construct logarithmic/geometric bins

    grid_min = 0.1 * loss_min
    grid_max = 10 * loss_max

    bin_edges = np.logspace(np.log10(grid_min), np.log10(grid_max), bin_n + 1)
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])  # geometric centers
    bin_widths = np.diff(bin_edges)

    m1ds = []

    # Calculate and plot first-moment densities

    for diagnostic in diagnostics:
        for epoch_i in range(len(diagnostic.epoch) - 1, -1, -epoch_skip):
            epoch = diagnostic.epoch[epoch_i]

            # Sample-resolved loss
            model_losses = diagnostic.per_sample_loss[epoch_i].to(device='cpu').numpy()

            # Histogram counts in logarithmic bins
            counts, _ = np.histogram(model_losses, bins=bin_edges, density=False)

            # Probability density per unit loss: dP/dL
            pdf = counts / (counts.sum() * bin_widths)

            # First-moment density per unit loss: L * dP/dL
            m1d = bin_centers * pdf

            # Convert to first-moment density per unit log10(L):
            # L * dP/dlog10(L) = ln(10) * L^2 * dP/dL
            if per_dlog10:
                m1d = np.log(10) * bin_centers * m1d

            m1ds.append(m1d)

            label = (
                diagnostic.model
                + ', '
                + diagnostic.optimizer
                + ', lr = '
                + str(diagnostic.learning_rate)
                + ', batch size = '
                + str(diagnostic.batch_size)
                + ', epoch = '
                + str(epoch)
            )

            if diagnostic.description:
                label = label + ', ' + diagnostic.description

            # Plot using bin centers
            ax[0, 0].plot(bin_centers, m1d, label=label)

    positive_m1d_mins = [m1d[m1d > 0].min() for m1d in m1ds if np.any(m1d > 0)]
    if positive_m1d_mins:
        m1d_min = min(positive_m1d_mins).item()
    else:
        m1d_min = 1e-12

    m1d_max = max([m1d.max() for m1d in m1ds]).item()

    ax[0, 0].legend(loc='best', fontsize=7, frameon=False)

    ax[0, 0].set_xlabel(loss_name)

    if per_dlog10:
        ax[0, 0].set_ylabel('First moment density\n' + loss_name + ' · dp/dlog10(' + loss_name + ')')
    else:
        ax[0, 0].set_ylabel('First moment density\n' + loss_name + ' · dp/d(' + loss_name + ')')

    if per_dlog10:
        ax[0, 0].set_xscale('log')
    else:
        if xlog:
            ax[0, 0].set_xscale('log')
        else:
            ax[0, 0].set_xscale('linear')

    if ylog:
        ax[0, 0].set_yscale('log')
    else:
        ax[0, 0].set_yscale('linear')

    if xlim:
        ax[0, 0].set_xlim(xlim)
    else:
        ax[0, 0].set_xlim([grid_min, grid_max])

    if ylim:
        ax[0, 0].set_ylim(ylim)
    else:
        ax[0, 0].set_ylim([m1d_min, m1d_max])

    # Increase all font sizes by a given factor
    for text in fig.findobj(match=lambda artist: hasattr(artist, 'get_fontsize')):
        text.set_fontsize(text.get_fontsize() * font_factor)

    if plot_file:
        plot_file = Path(plot_file)
        plot_dir = plot_file.parent
        os.makedirs(plot_dir, exist_ok=True)
        fig.savefig(plot_file, bbox_inches='tight')

        if verbose:
            print('Created plot', plot_file)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return
