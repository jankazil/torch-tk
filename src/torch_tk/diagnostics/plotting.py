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
    plot_file: Path = None,
    title: str = None,
    font_factor=1.5,
    figsize=(9, 6),
    xlim=None,
    ylim=None,
    loss_name='sqrt(loss)',
    pdf_bin_n=100,
    dpdlog10: bool = False,
    show_plot=True,
    verbose=True,
) -> None:
    '''
    Plot kernel-density estimates of square-root per-sample loss distributions.

    One or a list of Diagnostics objects are accepted. For each stored epoch, the
    function estimates a density curve on a logarithmic grid, plots it, and can
    optionally save the figure to disk and display it.
    '''

    diagnostics = diagnostics if isinstance(diagnostics, list) else [diagnostics]

    fig, ax = plt.subplots(figsize=figsize, nrows=1, ncols=1, squeeze=False)

    plt.suptitle(title, y=0.93, fontsize=8)

    for diagnostic in diagnostics:
        if diagnostic.per_sample_loss is None:
            raise ValueError('Each diagnostic must contain per-sample loss data.')
        if (diagnostic.per_sample_loss < 0).any():
            raise ValueError('Per-sample losses must be non-negative, as their square root is calculated.')

    # Construct PDF grid

    grid_min = min([diagnostic.per_sample_loss.min().sqrt().item() for diagnostic in diagnostics])
    grid_max = max([diagnostic.per_sample_loss.max().sqrt().item() for diagnostic in diagnostics])

    if grid_min <= 0:
        raise ValueError('Cannot construct logarithmic grid as loss contains at least one value <= 0.')

    grid = np.logspace(np.log10(grid_min), np.log10(grid_max), pdf_bin_n)

    pdfs = []

    for diagnostic in diagnostics:
        for n, epoch in enumerate(diagnostic.epoch):
            # Sample-resolved loss
            model_losses = diagnostic.per_sample_loss[n].to(device='cpu')

            # Semple-resolved square root of loss
            model_rlosses = model_losses.sqrt()

            # Construct the kernel density estimate
            kde = gaussian_kde(model_rlosses)

            # Evaluate the probability density on a logarithmic grid
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

    ax[0, 0].set_xscale('log')
    ax[0, 0].set_yscale('log')

    if xlim:
        ax[0, 0].set_xlim(xlim)
    else:
        ax[0, 0].set_xlim([grid_min, grid_max])
    if ylim:
        ax[0, 0].set_ylim(ylim)
    else:
        print(pdf_min, pdf_max)
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

    return plot_file
