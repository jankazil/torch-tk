'''
Plotting utilities for sample-resolved diagnostics.

This module provides a function to estimate and plot probability density curves
for the square root of per-sample loss values across one or more diagnostics
objects and training epochs.
'''

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from .diagnostics import Diagnostics


def plot_diagnostics(
    diagnostics: list[Diagnostics],
    plot_file: Path | str | None = None,
    title: str = None,
    loss_name='sqrt(loss)',
    font_factor=1.5,
    figsize=(9, 6),
    xlim=None,
    ylim=None,
    xlog=False,
    ylog=False,
    dpdlog10: bool = False,
    pdf_bin_n=100,
    pdf_log_grid=True,
    show_plot=True,
    verbose=True,
    epoch_skip=1,
) -> None:
    '''
    Plot kernel density estimates of square-rooted per-sample losses across epochs.

    This function accepts one Diagnostics object or a list of them. For each selected
    epoch in each diagnostic, it takes the per-sample losses for that epoch, computes
    their square roots, fits a Gaussian kernel density estimate, and plots the resulting
    density on a shared axis.

    The x-axis represents sqrt(loss) by default. The density can optionally be converted
    from dP/dsqrt(loss) to dP/dlog10(sqrt(loss)), which is useful for comparing mass
    across scales on a logarithmic x-axis.

    Args:
        diagnostics:
            A Diagnostics instance or a list of Diagnostics instances. Each diagnostic
            must provide per-sample loss data indexed by epoch.
        plot_file:
            Output path for saving the figure. If None, the figure is not written to disk.
        title:
            Figure title. If None, no meaningful title is added.
        loss_name:
            Label used for the x-axis quantity. Default is 'sqrt(loss)'.
        font_factor:
            Multiplicative scaling applied to all font sizes in the figure.
        figsize:
            Figure size passed to matplotlib.
        xlim:
            Optional x-axis limits.
        ylim:
            Optional y-axis limits.
        xlog:
            If True, use a logarithmic x-axis unless overridden by dpdlog10 handling.
        ylog:
            If True, use a logarithmic y-axis.
        dpdlog10:
            If True, plot density per unit log10 of the x quantity rather than per unit
            x. This also forces logarithmic scaling on the x-axis.
        pdf_bin_n:
            Number of grid points used to evaluate each KDE.
        pdf_log_grid:
            If True, evaluate the KDE on a logarithmically spaced grid when possible.
            If the minimum plotted value is non-positive, a linear grid is used instead.
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

    # Do your worst

    fig, ax = plt.subplots(figsize=figsize, nrows=1, ncols=1, squeeze=False)

    plt.suptitle(title, y=0.94, fontsize=8)

    for diagnostic in diagnostics:
        if diagnostic.per_sample_loss is None:
            raise ValueError('Each diagnostic must contain per-sample loss data.')
        if (diagnostic.per_sample_loss < 0).any():
            raise ValueError('Per-sample losses must be non-negative, as their square root is calculated.')

    # Construct PDF grid

    grid_min = min([diagnostic.per_sample_loss.min().sqrt().item() for diagnostic in diagnostics])
    grid_max = max([diagnostic.per_sample_loss.max().sqrt().item() for diagnostic in diagnostics])

    if grid_min <= 0:
        xlog = False
        pdf_log_grid = False

    if pdf_log_grid:
        grid = np.logspace(np.log10(grid_min), np.log10(grid_max), pdf_bin_n)
    else:
        grid = np.linspace(grid_min, grid_max, pdf_bin_n)

    pdfs = []

    for diagnostic in diagnostics:
        for epoch_i in range(len(diagnostic.epoch) - 1, -1, -epoch_skip):
            epoch = diagnostic.epoch[epoch_i]

            # Sample-resolved loss
            model_losses = diagnostic.per_sample_loss[epoch_i].to(device='cpu')

            # Semple-resolved square root of loss
            model_rlosses = model_losses.sqrt()

            # Construct the kernel density estimate
            kde = gaussian_kde(model_rlosses)

            # Evaluate the probability density on the grid
            pdf = kde(grid)

            # Convert to probability density per unit log10(L): dP/dlog10(sqrt(L)) = ln(10) * sqrt(L) * dP/dsqrt(L)
            # This will allow comparing values across scales when plotting sqrt(L) using a logarithmic grid.
            if dpdlog10:
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

    pdf_min = min([pdf[pdf > 0].min() for pdf in pdfs]).item()
    pdf_max = max([pdf[pdf > 0].max() for pdf in pdfs]).item()

    ax[0, 0].legend(loc='best', fontsize=7, frameon=False)

    ax[0, 0].set_xlabel(loss_name)

    if dpdlog10:
        ax[0, 0].set_ylabel('Probability density dp/dlog10(' + loss_name + ')')
    else:
        ax[0, 0].set_ylabel('Probability density dp/d' + loss_name + '')

    if dpdlog10:
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
