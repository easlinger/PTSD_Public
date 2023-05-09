#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=line-too-long
"""
Created on Wed Dec  9 11:10:37 2020.

@author: ena
"""

import functools
import numpy as np
import pandas as pd
import scipy as sp
import re
# import math
# import dcor
import statsmodels.api as sm
import seaborn as sb
import matplotlib
import matplotlib.pyplot as plt
from statsmodels.graphics.api import abline_plot
# from mpl_toolkits.axes_grid1 import make_axes_locatable, AxesGrid, ImageGrid
from functions_analysis import corr_dfs
from functions_data_cleaning import square_grid


def cap(string):
    """Capitalize the first letter of a string."""
    return string[0].upper() + string[1:]


def cap_title(string):
    """Capitalize 1st letter of each word (1st pasting if list, replacing '_' with ' '; retains all caps words)."""
    if type(string) == list:
        string = functools.reduce(lambda x, y: x + ' ' + y, string)
    string = re.sub('_', ' ', string)
    capped = [i if all([j.isupper() for j in i]) else i.title() for i in string.split(' ')]
    return functools.reduce(lambda x, y: x + ' ' + y, capped)


def square_heatmap(data_list, model_names, font={'family': 'serif', 'size': 10}, cmap='coolwarm'):
    """Make Heat Maps for Correlations, Regressions, or Factor Loadings."""
    matplotlib.rc('font', **font)
    fig, axes = plt.subplots(nrows=square_grid(len(data_list))[0], ncols=square_grid(len(data_list))[1],
                             figsize=[15, 10])
    low, high = min(0.6, min([f.min().min() for f in data_list])), 1  # minimum & maximum for heat map color bar
    for i in range(len(data_list)):
        df = data_list[i]
        for r in range(df.shape[0]):  # grey-out diagonal & upper triangle
            for c in range(df.shape[1]):
                if c >= r:
                    df.iloc[r, c] = np.nan
        _ = sb.heatmap(df.iloc[1:, :].astype(float), ax=axes.ravel().tolist()[i],
                       cmap=cmap, vmin=low, vmax=high, center=0, annot=True, fmt='.2f',
                       xticklabels=True, yticklabels=True)  # heat
        axes.ravel().tolist()[i].set_title(cap_title(model_names[i]))  # subplot title
    for a in range(len(axes.ravel().tolist())):  # ticks & spines
        ax = axes.ravel().tolist()[a]
        if a >= len(model_names):
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.tick_params(top=True, bottom=False, left=True, labeltop=True, labelbottom=False)  # labels: top
        plt.setp(ax.get_xticklabels(), rotation=-30, ha='right', rotation_mode='anchor')  # ticks
        ax.tick_params(which='minor', bottom=False, left=False)
        ax.grid(which='minor', color='w', linestyle='-', linewidth=3)
        for edge, spine in ax.spines.items():  # turn off spines & edges
            spine.set_visible(False)
    fig.tight_layout()
    return fig


def quick_corr_heatmap(df_1=None, df_2=None, df=None, subpop=None,
                       vars_1=None, vars_2=None, model=None, ax=None, **kwargs):
    """Create quick correlation heatmap for just two dataframes."""
    if model is not None:
        df = df[model]
    if subpop is not None:
        df = df[(df[subpop] == 1)]
    if df_1 is None:
        if type(vars_1) is dict:
            vars_1 = pd.Series(vars_1).explode()
        df_1 = df[vars_1]
    if df_2 is None:
        if type(vars_2) is dict:
            vars_2 = pd.Series(vars_2).explode()
        df_2 = df[vars_2]
    df_1.columns = [re.sub('_', ' ', c) for c in df_1.columns]
    df_2.columns = [re.sub('_', ' ', c) for c in df_2.columns]
    mat = corr_dfs(df_1, df_2)[1].applymap(lambda x: np.nan if x.count('*') < 2 else x)
    mat = mat.apply(lambda c: c.str.strip('*'), axis=1).astype(float)
    if ax is None:
        fig, ax = plt.subplots()
    sb.heatmap(mat, cmap='coolwarm', fmt='.2f', ax=ax, **kwargs)
    ax.tick_params(top=True, bottom=False, left=True, labeltop=True, labelbottom=False)  # labels: top
    plt.setp(ax.get_xticklabels(), rotation=-30, ha='right', rotation_mode='anchor', fontstyle='italic')  # ticks
    ax.tick_params(which='minor', bottom=False, left=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=3)
    plt.get_current_fig_manager().window.showMaximized()


def heatmaps(data, model_names, data_annotate=True,
             fig_title=None, title_fontsize=12, font={'family': 'serif', 'size': 10}, col_rename_dict={},
             annotation_kwargs={'fontsize': 10, 'fontstyle': 'italic', 'color': 'k', 'verticalalignment': 'center'},
             color_bar=True, cb_map='coolwarm', colorbar_center=0, cbar_min_range=[-0.3, 0.3],
             cbar_kwargs={'orientation': 'vertical', 'shrink': 1,
                          'extend': 'min', 'extendfrac': 0.1, 'drawedges': False},
             plot_ranges=False, sharex=False, sharey=False,
             intercors=None, intercors_label='',  # df for which to add subplot for its intercorrelations + label
             show=True, save=True):
    """Wrap heatmaps functions."""
    # matplotlib.rc('font', **font)
    from functions_and_classes import colorbar_min_range
    matplotlib.rcParams['mpl_toolkits.legacy_colorbar'] = False
    rr, cc = square_grid(len(model_names) + [0, 1][intercors is not None])  # row & columns (+ 1 for intercorrelations)
    dims = [cc, rr, 10, 26, 'landscape'] if len(data.columns) > 10 else [rr, cc, 20, 16, 'portrait']
    rows, cols, hh, ww, orient = dims  # (see above) wider or longer (more room for factor rows or DV columns)
    gs_kw_dict = None
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=[hh, ww], gridspec_kw=gs_kw_dict, sharex=sharex, sharey=sharey)  # subplots
    cmin, cmax = colorbar_min_range(data, cbar_min_range=cbar_min_range)  # color bar range
    tkvals = [round(i, 1) for i in np.arange(cmin - 0.1 * cmin, cmax + 0.1 * cmax, max([abs(cmin), abs(cmax)]) / 10)]
    if type(data_annotate) == bool:
        [data_annotate] * len(model_names)  # annotate T/F for every model (if T/F)
    # Iterate through Models (Subplots)
    for m in range(len(axes.ravel().tolist())):
        ax = axes.ravel().tolist()[m]  # mth subplot out of those not in last column (color bar column)
        if m < len(model_names):  # if within range of models to be plotted
            df = data.loc[model_names[m]].astype(float)
            da = data_annotate if type(data_annotate) == bool else data_annotate.loc[model_names[m]]
            if type(da) != bool:
                da.columns = [re.sub('_', ' ', b) for b in da.rename(col_rename_dict, axis=1).columns]
            dm = da if (type(da) != bool) else df
            cbar = True if m == len(model_names) - 1 else False
            # if 'orientation' in list(cbar_kwargs.keys()): # color bar below odd rows or last column
            #     rowm = ax.get_subplotspec().rowspan.start
            #     cbar = ax.is_last_col() if cbar_kwargs['orientation'] == 'vertical' else rowm%2 == 0
            _ = sb.heatmap(df, ax=ax, annot=da, fmt='.2f', annot_kws=annotation_kwargs,
                           # cbar_ax=axes.ravel().tolist()[-1],
                           cbar=cbar, cbar_kws={**cbar_kwargs, 'use_gridspec': True}, cmap=cb_map,
                           center=colorbar_center, vmin=min(tkvals), vmax=max(tkvals),
                           xticklabels=[cap_title(i) for i in dm.rename(col_rename_dict, axis=1).columns],
                           yticklabels=[cap_title(i) for i in dm.index.values])
            ax.set_title(re.sub('_', ' ', model_names[m]), fontsize=title_fontsize, fontdict={'fontweight': 40})
            ax.tick_params(top=True, bottom=False, left=True, labeltop=True, labelbottom=False)  # labels: top
            plt.setp(ax.get_xticklabels(), rotation=-30, ha='right',
                     rotation_mode='anchor', fontstyle='italic')  # ticks
            ax.tick_params(which='minor', bottom=False, left=True)
            ax.grid(which='minor', color='w', linestyle='-', linewidth=3)
        elif (intercors is not None) or (plot_ranges):  # plot histograms for inter-correlations or correlation ranges
            # fr = [pd.concat([q.corr().unstack().loc[f].drop(f) for f in q.columns], keys=q.columns) for q in intercors]
            extra_dfs = intercors if intercors is not None else [data.loc[m].astype(float) for m in model_names]
            if intercors is not None:
                for i, q in enumerate(extra_dfs):
                    if intercors is not None:
                        frs = pd.concat([q.corr().unstack().loc[f].drop(f) for f in q.columns], keys=q.columns)
                    else:
                        frs = q.unstack()
                    b = col_rename_dict[model_names[i]] if model_names[i] in col_rename_dict.keys() else model_names[i]
                    _ = sb.distplot(frs, label=re.sub('_', ' ', b), ax=ax, kde_kws={'cut': 0})
            else:
                hist_range = [min(abs(data).min()), max(abs(data).max())]
                bins = [[q[0] * q[1] for q in zip(max([abs(x) for x in i]))] for i in zip([0, 0.3], hist_range)]
                sb.histplot(x=abs(data.stack()).rename_axis(['Model', 'Factor', 'Outcome']), ax=ax,
                            data=data.stack().rename_axis(['Model', 'Factor', 'Outcome']), hue='Factor',
                            binrange=bins)
            if intercors is not None:
                ax.legend()
            ax.set_title(str('%s Correlation Magnitudes' % intercors_label).strip())
        else:
            if not sharey:
                ax.set_yticks([])
            if not sharex:
                ax.set_xticks([])
            ax.grid(b=False)
        if not sharex and not sharey:
            for edge, spine in ax.spines.items():  # turn off spines & edges
                spine.set_visible(False)
    # Save & Display
    plt.subplots_adjust(left=0.03, bottom=0.12, top=0.85, right=0.97, hspace=0.15, wspace=0.15)  # adjust
    fig.tight_layout()
    if show:
        plt.get_current_fig_manager().window.showMaximized()  # maximize plot window
        fig.show()
    return fig


def kde_3d_setup(data, xvar, yvar):
    """
    3-D Kernel Density Plot.

    (from towardsdatascience.com/simple-example-of-2d-density-plots-in-python-83b83b934f67)
    """
    data = data.dropna()  # drop missing (ENA; 12/2020)
    x = data[xvar]
    y = data[yvar]  # define borders
    deltaX = (max(x) - min(x)) / 10
    deltaY = (max(y) - min(y)) / 10
    xmin = min(x) - deltaX
    xmax = max(x) + deltaX
    ymin = min(y) - deltaY
    ymax = max(y) + deltaY
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]  # mesh grid
    # Fit Gaussian Kernel
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = sp.stats.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)
    # ax.view_init(60, 35)
    return [xx, yy, f]


def plot_kde(df, DVs, factor, model):
    """Plot kernel density estimate."""
    rows, cols = square_grid(len(DVs))  # row & columns (as close to square as possible)
    fig, axes = plt.subplots()
    fig.set_frameon(False)
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['bottom'].set_visible(False)
    axes.spines['left'].set_visible(False)
    for i in [k for k in axes.spines]:
        axes.spines[i].set_visible(False)
    for y in range(len(DVs)):
        ax = fig.add_subplot(rows, cols, y + 1, projection='3d')
        xx, yy, f = kde_3d_setup(df, DVs[y], factor)
        _ = ax.plot_surface(xx, yy, f, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')
        ax.set_xlabel(DVs[y])
        ax.set_ylabel(factor)
        ax.set_zlabel('PDF')
        ax.set_title(DVs[y])
        ax.set_ylim(max(df[factor]), min(df[factor]))  # invert axes so higher scores at the front
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['bottom'].set_visible(False)
    axes.spines['left'].set_visible(False)
    axes.set_xticks([])
    axes.set_yticks([])
    # fig.colorbar(surf, shrink=0.5, aspect=5) # add color bar indicating PDF
    fig.suptitle('Gaussian Kernel Density Estimation of %s %s Factor and Outcomes' % (model, factor))
    fig.tight_layout()
    plt.subplots_adjust(left=0.05, bottom=0.05, top=0.88, right=0.95, hspace=0.1)  # adjust margins
    return fig


# def plot_dist_corr(data):
#     """
#     Distance Correlations (doesn't assume linearity).

#     From mycarta.wordpress.com/2019/04/10/data-exploration-in-python-distance-correlation-and-variable-clustering/
#     """
#     def dist_corr(X, Y, pval=True, nruns=2000):
#         dc = dcor.distance_correlation(X, Y)
#         pv = dcor.independence.distance_covariance_test(X, Y, exponent=1.0, num_resamples=nruns)[0]
#         if pval:
#             return (dc, pv)
#         else:
#             return dc

#     def corrfunc(x, y, **kws):
#         d, p = dist_corr(x, y)
#         if p > 0.01:
#             pclr = 'Darkgray'
#         else:
#             pclr = 'Darkblue'
#         ax = plt.gca()
#         ax.annotate("DC = {:.2f}".format(d), xy=(.1, 0.99), xycoords=ax.transAxes, color=pclr, fontsize=10)
#     g = sb.PairGrid(data, diag_sharey=False)
#     # axes = g.axes
#     g.map_upper(plt.scatter, linewidths=1, edgecolor="w", s=90, alpha=0.5)
#     # g.map_upper(corrfunc)
#     g.map_diag(sb.kdeplot, lw=4, legend=False)
#     g.map_lower(sb.kdeplot, cmap="Blues_d")
#     g.map_upper(sb.kdeplot, cmap="Blues_d")
#     plt.show()


def plot_regression(y, fit, logistic=False):
    """Regression Plots (from https://www.statsmodels.org/stable/examples/notebooks/generated/glm.html)."""
    fig, axes = plt.subplots(nrows=1, ncols=2)
    yhat = fit.fittedvalues
    axes[0].scatter(yhat, y)
    line_fit = sm.OLS(y, sm.add_constant(yhat, prepend=True)).fit()
    abline_plot(model_results=line_fit, ax=axes[0])
    axes[0].set_title('Model Fit Plot')
    axes[0].set_ylabel('Observed Values')
    axes[0].set_xlabel('Fitted Values')
    axes[1].scatter(yhat, fit.resid_pearson)
    axes[1].hlines(0, 0, 1)
    axes[1].set_xlim(0, 1)
    axes[1].set_title('Residual Dependence Plot')
    axes[1].set_ylabel('Pearson Residuals')
    axes[1].set_xlabel('Fitted values')
