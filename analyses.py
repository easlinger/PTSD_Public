#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=line-too-long
"""
Created on Thu Nov 19 13:06:33 2020.

@author: ena
"""

# %% Set Up

# Imports & Custom Scripts
import json
import re
import os
import functools
import warnings
import copy
from docx.shared import Inches
import statsmodels.api as sm
import scipy as sp
import rpy2.robjects as ro
import matplotlib.pyplot as plt
# from sklearn.decomposition import FactorAnalysis
import seaborn as sb
import pandas as pd
import numpy as np
from functions_documentation import string_list
from functions_visualization import quick_corr_heatmap
from functions_data_cleaning import square_grid
from functions_data_cleaning import jsonKeys2int, jsonKeys2int_nested, wh, dict_part
from functions_and_classes import (corr_dfs, Factor_Model_Results,
                                   # Factor_Model, Factor_Model_Analysis,
                                   model_dictionary_to_content)
# from functions_and_classes import Factor_Model_Results


# Options
pd.options.display.max_columns = 20
pd.options.display.max_rows = 100
pd.options.display.latex.longtable = True
write = True  # write MPlus syntax?
BAYES = True
# From https://github.com/ejolly
os.environ[
    "KMP_DUPLICATE_LIB_OK"
] = "True"  # Recent versions of rpy2 sometimes cause the python kernel to die when running R code; this handles that

# Dictionaries
with open('Dictionaries/arguments.json') as file:
    args = json.load(file)
with open('Dictionaries/variables_n3_check_a1.json') as file:  # checks for frequencies in overall N-III sample
    check_n3 = json.load(file, object_hook=jsonKeys2int_nested)
with open('Dictionaries/variables_n2_check_a1.json') as file:  # checks for symptom frequencies in trauma-exposed N-II sample
    check_n2 = json.load(file, object_hook=jsonKeys2int_nested)
with open('Dictionaries/variables_n3_recode_from_mplus.json') as file:
    variables_n3_recode = json.load(file, object_hook=jsonKeys2int)
with open('Dictionaries/dataset_files.json') as file:
    data_files = json.load(file)
with open('Dictionaries/variables_n2.json') as file:
    n2_dict_rename = json.load(file)
with open('Dictionaries/dataset_columns.json') as file:  # dictionaries with NESARC-II & -III column names
    column_names = json.load(file)
with open('Dictionaries/variables_n2_types.json') as file:  # dictionaries for variable data types
    n2_dict_types = json.load(file)
with open('Dictionaries/models_all.json') as file:  # dictionaries for variable data types
    models_all_dict = json.load(file)
n2_types = dict(pd.Series(dict([(n2_dict_rename[k], n2_dict_types[k] if k in n2_dict_types else np.nan)
                                for k in n2_dict_rename])).dropna())
n2_types.update({'Panic': 'int'})
# n2_sx_all = ['B1', 'B2', 'B3a', 'B3b', 'B4', 'B5', 'C1', 'C2a', 'C2b', 'D1', 'D2', 'D5', 'D6', 'D7', 'E1', 'E3', 'E4', 'E5', 'E6']
n2_sx = ['B1', 'B2', 'B3', 'B4', 'B5', 'C1', 'C2', 'D1', 'D2', 'D5', 'D6', 'D7', 'E1', 'E3', 'E4', 'E5', 'E6']

# Data Files
data_file_original = data_files['nesarc3_original']  # main file
data_file_id = data_files['nesarc3_id']  # has IDs, clustering, stratification, & weight
data_file_original_n2 = data_files['nesarc2_original']
data_file = data_files['nesarc3_cleaned']
data_file_n2 = data_files['nesarc2_cleaned']

# Arguments
args_dicts = [args[a] for a in ['create', 'n3', 'fac_series_dict', 'cfas', 'cfas_PTSD', 'bfs', 'rerun_r']]
create, args_n3, fac_series_dict, args_cfas, args_cfas_PTSD, args_bfs, rerun_r = args_dicts
args_n3.update({'recode_map': variables_n3_recode})
model_names, fac_series = list(fac_series_dict.keys()), pd.Series(fac_series_dict)  # model names & factors
for i, d in enumerate([args_cfas, args_cfas_PTSD, args_bfs]):
    ch_dict = None if i == 1 else check_n3
    d.update({'model_names': model_names, 'fac_series': fac_series, 'check_dict': ch_dict, 'write': write, **args_n3})
impair = ['Interfere_Daily_Life', 'Harder_Daily_Activities', 'Relationships_Problems', 'Problems_Work_School']
# args_cfas_PTSD['DV_subsets_dictionary'].update({'Impairment': impair})
args_cfas_PTSD.update({'join_data': False})
ix_vars = ['Upset', 'Distress', 'Interfere_Daily_Life', 'Harder_Daily_Activities', 'Relationships_Problems', 'Problems_Work_School']
args_cfas_PTSD['DV_subsets_dictionary'].update({'Impairment': ix_vars})
args_cfas_all = copy.deepcopy(args_cfas)
args_cfas_n2 = copy.deepcopy(args_cfas)
args_cfas_PTSD_n2 = copy.deepcopy(args_cfas_PTSD)
args_bfs_n2 = copy.deepcopy(args_bfs)
args_cfas_all.update({'model_dictionary': models_all_dict, 'indicator_dictionary': args_cfas['indicator_items'],
                      'file_suffix': 'all', 'caption_suffix': ' (NESARC-III: All Indicators)'})  # with sub-items
n2_ind_dict = dict(zip(n2_sx, [args_cfas['indicator_dictionary'][k] for k in n2_sx]))  # N-II indicator dictionary
model_dict_n2 = dict(**pd.DataFrame(args_cfas['model_dictionary'])[n2_sx].rename(
    dict(zip(np.arange(len(args_cfas['model_dictionary'])), model_names))).T.apply(
        lambda x: list(x), axis=1))  # model dictionary with N-II symptoms only
n2_update_dict = {'data_file': data_file_n2, 'data_file_mplus': data_files['nesarc2_mplus'],
                  'check_dict': check_n2, 'data_columns': column_names[data_file_n2], 'data_types': n2_types,
                  'file_suffix': 'n2', 'caption_suffix': ' (NESARC-II)',
                  'model_dictionary': model_dict_n2, 'indicator_dictionary': n2_ind_dict}
mod_dict_cfas_all_n2 = dict(list(pd.Series([(c, args_cfas_all['model_dictionary'][c]) if c in column_names[data_file_n2] else np.nan
                                            for c in args_cfas_all['model_dictionary']]).dropna()))
n2_all_ind_dict = dict(zip(mod_dict_cfas_all_n2.keys(), [args_cfas_all['indicator_dictionary'][k] for k in mod_dict_cfas_all_n2]))
args_cfas_n2.update(n2_update_dict)
args_cfas_PTSD_n2.update(n2_update_dict)
args_cfas_PTSD_n2.update({'file_suffix': 'PTSD_n2', 'caption_suffix': ' (NESARC-II PTSD Sample)'})
args_bfs_n2.update(n2_update_dict)
args_cfas_all_n2 = copy.deepcopy(args_cfas_n2)
args_cfas_all_n2.update({'file_suffix': 'all_n2', 'caption_suffix': ' (NESARC-II: All Indicators)',
                         **dict(zip(['model_dictionary', 'indicator_dictionary'],
                                    [mod_dict_cfas_all_n2, n2_all_ind_dict]))})
# args_cfas_PTSD.update({'binary': [False, True, False, True]})

# Documentation
man_dir = data_files['manuscript_directory']
secs_main = ['Title', 'Abstract', 'Introduction', 'Methods', 'Discussion', 'References']
secs_out = ['Title', 'Abstract', 'Introduction', 'Methods', 'Results', 'Discussion', 'References', 'Tables', 'Figures']
files_main = [man_dir + '/%s.docx' % s for s in secs_main]
sec_s = ['Methods', 'Results']
files_supplement = [man_dir + '/Supplement_%s.docx' % s for s in sec_s]
doc_args = {'directory': 'Manuscripts', 'source_files': files_main, 'sections': secs_main,
            'source_files_supplement': files_supplement, 'output_sections': secs_out,
            'title': 'Validating Competing Structures of Post-Traumatic Stress Disorder',
            'sections_supplement': sec_s, 'sections_supplement': sec_s, 'output_sections_supp': sec_s + ['Tables']}

# Options
clear_compare = True  # clear past analyses comparisons attributes upon re-running next cell
sh = False  # don't show certain plots in window

# Attributes of Interest
comp_attrs = functools.reduce(
    lambda i, j: i + j, [[f'correlations{s}', f'regressions{s}'] for s in ['', '_p']])  # to compare cfas, cfas_PTSD, bfs
res = [f'results_{x}' for x in comp_attrs]
attrs_xl = res + ['results_regressions_tables', 'results_fit']
attrs_wd = ['results_fit'] + ['factor_loadings', 'Factor_Correlations'] + [res[2], re.sub('_p', '_cis', res[1])]
comp_attrs = functools.reduce(lambda x, y: x + y, res)
# types = ['descriptives_factors', 'results_correlations', 'results_regressions'] # heat map types
types = ['results_correlations', 'results_regressions']  # heat map types

# Argument Tweaking
args_cfas_n2.update({'ESTIMATOR': 'MLR', 'TYPE': 'COMPLEX',
                     # 'stratification': None, 'cluster': None, 'weight': None,
                     'join_data': True})
args_cfas_n2['descriptives_groups'] = 'SEX'
model_content, _, _, _, _ = model_dictionary_to_content(args_cfas_n2['model_names'], args_cfas_n2['model_dictionary'])
_ = args_cfas_n2.pop('model_names')
_ = args_cfas_n2.pop('fac_series')
_ = args_cfas_n2.pop('model_dictionary')
# model_content = model_content.drop(['Externalizing', 'Anhedonia', 'Hybrid'], axis=1)
args_cfas_n2.update(dict(model_content=model_content))
args_cfas_n2['DV_subsets_dictionary']['Diagnoses'].remove('CD')
args_cfas_PTSD_n2.update({'ESTIMATOR': 'MLR', 'TYPE': 'COMPLEX',
                          # 'stratification': None, 'cluster': None, 'weight': None,
                          'join_data': True})

# Data Checks
# nesarc3 = type_convert(pd.read_csv(args_cfas['data_file'], index_col=False), args_cfas['data_types'])
# nesarc2 = type_convert(pd.read_csv(args_cfas_n2['data_file'], index_col=False), args_cfas_n2['data_types'])
# check_n3_a1, check_n3_a1_mismatch = check_frequencies(nesarc3, check_n3, plot=False)  # check sx frequencies
# check_n2_a1, check_n2_a1_mismatch = check_frequencies(nesarc2, check_n2, plot=False)  # check sx frequencies


# %% Create Models

# Example Command (Terminal) to Transfer & Run CFA Syntax in the Background on MandelBot
# sftp SeanLab@10.160.112.184 -y
# cd Factor_Analysis
# put MPlus/*.inp MPlus  ## ...then quit & enter:
# put MPlus/*.sh MPlus  ## ...then quit & enter:
# ssh -f SeanLab@10.160.112.184 -y
# cd Factor_Analysis/MPlus
# chmod +x batch_CFA.sh
# source batch_CFA.sh
# cat DSM_CFA_log.txt && cat DSM_CFA_log.txt  # see .sh file for names of .txt files it's writing to

# Example of how to create from scratch (the NESARC-III chunk below if for ones that already have Mplus results)
# if 'cfas' in create:
#     cfas_n2 = Factor_Model(**args_cfas_n2)
#     cfas_n2.create_syntax(**args_cfas_n2)
#     _ = input('Please run Mplus syntax before proceeding.\n')
#     cfas_n2.read_mplus(join_data=args_cfas_n2['join_data'])  # read output

# Models
syntax = input('Read past MPlus input (enter) or re-write (w)? ')
syntax = 'create' if syntax == 'w' else 'read'
models = []
all_indicators = False
create = []
if input('Include CFAs (enter) or no (n): ').strip() == '':
    create += ['cfas']
if input('Include PTSD subset (enter) or no (n): ').strip() == '':
    create += ['ptsd']
if input('Include bifactor (enter) or no (n): ').strip() == '':
    create += ['bfs']
run_n3 = input('Include NESARC-III models (enter) or no (n): ').strip() == ''
run_n2 = input('Include NESARC-II models (enter) or no (n): ').strip() == ''
if 'cfas' in create:
    if all_indicators:
        if run_n3:
            cfas_all = Factor_Model_Results(**args_cfas_all, syntax=syntax)
            models += [cfas_all]
        if run_n2:
            cfas_all_n2 = Factor_Model_Results(**args_cfas_all_n2, syntax=syntax)
            models += [cfas_all_n2]
    else:
        if run_n3:
            cfas = Factor_Model_Results(**args_cfas, syntax=syntax)
            models += [cfas]
        if run_n2:
            cfas_n2 = Factor_Model_Results(**args_cfas_n2, syntax=syntax)
            models += [cfas_n2]
# if 'cfas_wlsmv' in create:
    # cfas_wlsmv = Factor_Model_Analysis(**args_cfas_wlsmv, **args_n3)
if 'ptsd' in create:
    if run_n3:
        cfas_PTSD = Factor_Model_Results(**args_cfas_PTSD, syntax=syntax)
        models += [cfas_PTSD]
    if run_n2:
        cfas_PTSD_n2 = Factor_Model_Results(**args_cfas_PTSD_n2, syntax=syntax)
        models += [cfas_PTSD_n2]
if 'bfs' in create:
    if run_n3:
        bfs = Factor_Model_Results(**args_bfs, syntax=syntax)
        models += [bfs]
    if run_n2:
        bfs_n2 = Factor_Model_Results(**args_bfs_n2, syntax=syntax)
        models += [bfs_n2]

# Data Checks
# cfas_df = type_convert(cfas.data_original, args_cfas['data_types'])
# check_n3_cfas, check_n3_cfas_mismatch = check_frequencies(cfas_df, check_n3, plot=False)  # check sx frequencies

# Get Attributes of Different Types
if 'cfas' in create:
    if run_n3:
        des_cfas, res_cfas = cfas.get_attributes_from_stems(['descriptives', 'results'])
    if run_n2:
        des_cfas_n2, res_cfas_n2 = cfas_n2.get_attributes_from_stems(['descriptives', 'results'])
if 'ptsd' in create:
    if run_n3:
        des_cfas_PTSD, res_cfas_PTSD = cfas_PTSD.get_attributes_from_stems(['descriptives', 'results'])
    if run_n2:
        des_cfas_n2_PTSD, res_cfas_n2_PTSD = cfas_PTSD_n2.get_attributes_from_stems(['descriptives', 'results'])
if 'bfs' in create:
    if run_n3:
        des_bfs, res_bfs = bfs.get_attributes_from_stems(['descriptives', 'results'])
    if run_n2:
        des_bfs_n2, res_bfs_n2 = bfs_n2.get_attributes_from_stems(['descriptives', 'results'])


# %% Check Data

# Variables
mod = models[0]
sxs, y = list(mod.model_content.index.values), pd.Series(dict_part(mod.DV_subsets_dictionary, 'items'))
all_dvs, all_vars = y.explode(), list(y.explode()) + sxs
dv_facs = mod.fac_content.apply(lambda x: list(x.keys()) + list(all_dvs))
df_o = mod.data_original.set_index(mod.data_info['index_col'])[all_vars].replace(mod.data_info['na_flag'], np.nan)

# Symptom Counts & Correlations with/without Asymptomatic People
sx_cts = mod.data['DSM'][sxs + ['PTSD']].set_index('PTSD', append=True).T.sum()  # symptom counts
if sx_cts.reset_index().PTSD.sum() != 2339:
    warnings.warn('PTSD diagnoses dummy-coding does not match expect number (2,339).')
sx_cts.groupby('PTSD').apply(lambda x: f'{100*np.mean(x != 0).round(2)}% (N = {sum(x != 0)})')  # >= 1 sx
sb.displot(data=sx_cts.to_frame(name='Symptom Count'), x='Symptom Count', hue='PTSD')  # barplot
print('\n\nSX counts:\n\n', sx_cts.describe(), '\n\n', sx_cts.groupby('PTSD').describe())
no_zeros_ixs = (sx_cts == 0).replace(True, np.nan).dropna().reset_index('PTSD').index.values
no_zeros_corr, all_ids_corr = [dv_facs.to_frame(name='vars').apply(
    lambda x: mod.data[x.name].loc[i][x.vars].corr().loc[mod.fac_content[x.name].keys()][all_dvs],
    axis=1) for i in [no_zeros_ixs, mod.data['DSM'].index.values]]  # correlations DSM; no 0 sx IDs & full sample
no_zeros_corr = pd.concat(list(no_zeros_corr), keys=no_zeros_corr.index.values, names=['Model', 'Factor'])
all_ids_corr = pd.concat(list(all_ids_corr), keys=all_ids_corr.index.values, names=['Model', 'Factor'])
diff_no_zeros = all_ids_corr.drop('PTSD', axis=1) - no_zeros_corr.drop('PTSD', axis=1)  # (a)sx correlation differences
comp_no_zeros = pd.concat([no_zeros_corr.drop('PTSD', axis=1), all_ids_corr.drop('PTSD', axis=1)],
                          keys=['Symptomatic', 'Full Sample'], names=['Sample', 'Model', 'Factor'])
for w in [no_zeros_corr, diff_no_zeros]:
    fig, axes = plt.subplots(*square_grid(len(all_ids_corr.reset_index().Model.unique())))
    for i, m in enumerate(no_zeros_corr.reset_index().Model.unique()):
        sb.heatmap(data=w.loc[y].round(2), cmap='coolwarm', center=0, annot=False, ax=axes.ravel()[i])
    fig.tight_layout()
comp_nz_long = comp_no_zeros.stack().rename_axis(comp_no_zeros.index.names + ['Outcome']).to_frame(name='Correlation')
diff_nz_long = diff_no_zeros.stack().rename_axis(diff_no_zeros.index.names + ['Outcome']).to_frame(name='Correlation')
sb.catplot(data=comp_nz_long.reset_index(), x='Outcome', y='Correlation', hue='Sample', kind='bar', palette='Blues',
           col='Model', col_wrap=4)
sb.displot(data=diff_nz_long, x='Correlation', hue='Factor',
           col='Model', col_wrap=4, kind='kde', fill=True)  # differences differ by factor?
sb.displot(data=diff_nz_long, x='Correlation', hue='Outcome',
           col='Model', col_wrap=4, kind='kde', fill=True)  # differences differ by outcome?


# Check MPlus Saved Data
for i, o in enumerate(models):
    mod_obj = ['CFAs', 'CFAs PTSD', 'Bifactor'][i]
    df_c = df_o if i != 1 else df_o[df_o.PTSD == 1]  # subset by PTSD if needed
    ch_o = pd.concat([o.data[d][all_vars].compare(df_c) for d in o.data])
    string = '\n\n\n{"=" * 80}\n{mod_obj} GREAT! Original & save data\n{"=" * 80}'
    if ch_o.empty:
        print(re.sub('save data', 'save data MATCH', string))
    else:
        warnings.warn(re.sub('GREAT', 'MISMATCH', string))
        print(ch_o)

# cfas_PTSD versus cfas PTSD Sub-Sample
ch_sx = [cfas_PTSD.data[m][sxs].compare(mod.data[m][mod.data[m].PTSD == 1][sxs]) for m in model_names]
ch_y = [cfas_PTSD.data[m][y.explode()].compare(mod.data[m][mod.data[m].PTSD == 1][y.explode()]) for m in model_names]
for i, ch in enumerate([ch_sx, ch_y]):
    if all([d.empty for d in ch]):
        obj = ['Symptoms', 'DV'][i]
        print('\n\n\n%s\n%s in objects for full & PTSD samples match!\n%s' % ('=' * 80, obj, '=' * 80))
    else:
        mis = wh([d.empty for d in ch], False, list([sxs, y.explode()][i]))
        warnings.warn('\n\n\n%s\nMismatch in full & PTSD samples: %s\n%s' % ('=' * 80, string_list(mis), '=' * 80))

# Check Correlation of PTSD Sub-Sample-Estimated Factor Scores & Subsetted Full-Sample Factor Scores
print(f'\n\n{"=" * 80}\nFactor Correlations as Estimated in the PTSD Sample & Subset of Full Sample\n{"=" * 80}\n')
df_sub = [mod.data[m][mod.data[m].PTSD == 1][fac_series.loc[m]] for m in mod.model_names]
df_ptsd = [cfas_PTSD.data[m][fac_series.loc[m]] for m in cfas_PTSD.model_names]
fcorrs = [corr_dfs(*m)[0].round(2) for m in zip(df_sub, df_ptsd)]
[print(mod.model_names[m], dict(zip(fac_series[m], np.diag(fcorrs[m])))) for m in range(len(mod.model_names))]
for dvs in dict_part(mod.DV_subsets_dictionary, 'items'):
    fig, axes = plt.subplots(*square_grid(len(mod.model_names)))
    fig_2, axes_2 = plt.subplots(*square_grid(len(mod.model_names)))
    diff = []
    for i, m in enumerate(mod.model_names):
        cbar = True if m == mod.model_names[-1] else False
        quick_corr_heatmap(df=mod.data[m], subpop='PTSD', ax=axes.ravel()[wh(mod.model_names, m)],
                           annot=True, vars_1=mod.fac_series[m], vars_2=dvs, xticklabels=True, yticklabels=True,
                           cbar=cbar, center=0)  # subset of full sample-estimated factor score correlations with DVs
        quick_corr_heatmap(df=cfas_PTSD.data[m], subpop='PTSD', ax=axes_2.ravel()[wh(cfas_PTSD.model_names, m)],
                           annot=True, vars_1=cfas_PTSD.fac_series[m], vars_2=dvs, xticklabels=True, yticklabels=True,
                           cbar=cbar, center=0)  # PTSD-estimated factor score correlations with DVs
        r_subset = corr_dfs(mod.data[m][mod.data[m].PTSD == 1][mod.fac_series[m]],
                            mod.data[m][mod.data[m].PTSD == 1][y.explode()])
        r_PTSD = corr_dfs(cfas_PTSD.data[m][cfas_PTSD.fac_series[m]], cfas_PTSD.data[m][y.explode()])
        diff = diff + [r_subset[0] - r_PTSD[0]]
        axes.ravel()[wh(mod.model_names, m)].set_title(m)
        axes_2.ravel()[wh(mod.model_names, m)].set_title(m)
    for edge, spine in axes.ravel()[len(mod.model_names)].spines.items():  # turn off spines & edges
        spine.set_visible(False)
    axes.ravel()[len(mod.model_names)].set_xticks([])
    axes.ravel()[len(mod.model_names)].set_yticks([])
    fig_diff, axes_diff = plt.subplots(figsize=[35, 20])
    sb.heatmap(pd.concat(diff, keys=mod.model_names).T, annot=True, ax=axes_diff,
               cbar=None, cmap='coolwarm', center=0)
    fig.suptitle('DV Correlations with Full Sample-Estimated Factor Scores (Subsetted by PTSD)')
    fig_2.suptitle('DV Correlations with PTSD Sample-Estimated Factor Scores')
    fig_diff.suptitle('Correlation Differences: PTSD Sample-Estimated versus Subset of Full Sample-Estimated Factors')
    fig.show()
print('\n\n')


# %% Explore Differences

# Correlations Differences
cr = [pd.concat([o.results_correlations_p[k] for k in cfas.results_correlations_p], axis=1) for o in [cfas, cfas_PTSD]]
sn = 1 if cfas.alpha >= 0.05 else 2 if cfas.alpha >= 0.01 else 3
all_cor = pd.concat([k.applymap(lambda x: float(x.strip('*')) if x.count('*') >= sn else 0) for k in cr],
                    keys=['Full', 'PTSD'], names=['Sample', 'Model', 'Factor'])
cfs = {'DA': ['CM', 'AA'], 'An': ['Ne', 'Nu'], 'EB': ['HA', 'DA']}
diffs = [dict(zip(cfs[k], [abs(all_cor.loc[:, :, k, :] - all_cor.loc[:, :, x, :]) for x in cfs[k]])) for k in cfs]
cms = [pd.concat(d, keys=d.keys(), names=['F', 'S', 'M']).dropna(how='all').reset_index(2, drop=1) for d in diffs]
zpd = zip(cfs.keys(), cms)
z = pd.DataFrame(dict([(c[0], c[1].reset_index().groupby(['F', 'S']).max().T.max().rename_axis([''] * 2))
                       for c in zpd]))
print('\n\nMaximum Differences (N.S. = 0): \n\n', z.stack().unstack(1))

# Externalizing Factor
for x in ['EB', ['EB', 'In', 'Av']]:  # try comparisons with all other factors & EB versus EB & non-In/Av factors
    if x != 'EB':
        print('\n\n Excluding In & Av factors from comparison: \n\n')
    dx_ext_non_eb = cfas.results_correlations['Diagnoses'].drop(x, axis=0, level=1)[['AUD', 'NUD', 'CD', 'ASPD']]
    dx_ext_eb = cfas.results_correlations['Diagnoses'].loc[:, 'EB', :][['AUD', 'NUD', 'CD', 'ASPD']]
    for m in dx_ext_eb.index.values:  # EXT dx correlations: EB vs. others (maximum difference)
        [print(c, '{:.3f}'.format((dx_ext_eb.loc[m][c] - dx_ext_non_eb[c]).max())) for c in dx_ext_eb.columns]

# Negative Affect/Anhedonia
sig = '*' if cfas.alpha >= 0.05 else '**' if cfas.alpha >= 0.01 else '***'
for i, r in enumerate([cfas.results_correlations_p, cfas_PTSD.results_correlations_p]):
    cors = [copy.deepcopy(r[k]) for k in r]
    cors = [k.applymap(lambda x: np.nan if x.count('*') < sig.count('*') else float(x.strip('*'))) for k in cors]
    head = ('=' * 80, ['Full Sample', 'PTSD Sub-Sample'][i], '=' * 80)
    print('\n\n%s\n\nAnhedonia - Negative Affect Correlations: %s\n\n%s\n\n' % head)
    [print(k.loc[:, 'An', :] - k.loc[:, 'Ne', :]) for k in cors]
    print('\n\n----\n\nMaximum Differences: \n')
    [print(abs(k.loc[:, 'An', :] - k.loc[:, 'Ne', :]).max().replace(np.nan, 'NS')) for k in cors]
    print('\n\n----\n\nFactor Correlations: \n')
    fcors = [cfas, cfas_PTSD][i].descriptives_factors['correlations']['Within']
    print(fcors.loc[:, ['Ne', 'An'], :][['Ne', 'An']].describe())


# %% Comorbidities

# Without Comorbidities
dvs_nd = cfas.DV_subsets_dictionary['SF-12'] + cfas.DV_subsets_dictionary['Drinking']
i_nc = cfas.data['DSM'][cfas.data['DSM'][cfas.DV_subsets_dictionary['Diagnoses'][1:]].apply(sum, axis=1) == 0].index
i_nc_ptsd = i_nc.intersection(cfas_PTSD.data['DSM'].index)
corrs_nc = corr_dfs(cfas.factor_scores.loc[i_nc.values], cfas.data['DSM'].loc[i_nc.values][dvs_nd])
corrs_nc_ptsd = corr_dfs(cfas_PTSD.factor_scores.loc[i_nc_ptsd], cfas_PTSD.data['DSM'].loc[i_nc_ptsd][dvs_nd])

# Comorbidities Predicting GF
xx = sm.add_constant(bfs.data['DSM'][cfas.DV_subsets_dictionary['Diagnoses'][1:]])
yy = bfs.data['DSM']['GF']
tmp = sm.OLS(yy, xx)

# Partialling Out Comorbidities
xx = sm.add_constant(cfas.data['DSM'][['BPD'] + list(cfas.fac_series['DSM'])])
yy = bfs.data['DSM'][cfas.DV_subsets_dictionary['SF-12'][0]]
tmp = sm.OLS(yy, xx)


# %% Print Regression Results by Groups of Factors

stack = False
distress = ['Depressed', 'MDD', 'GAD', 'Calm']
numb = ['Depressed', 'MDD', 'Less_Accomplished']
fear = ['Panic', 'Phobia']
externalizing = ['AUD', 'NUD', 'BPD', 'ASPD', 'CD', 'Less_Careful'] + cfas.DV_subsets_dictionary['Drinking']
f_grps = [['Nu', 'Ne', 'DA'], ['AA'], ['An', 'Nu'], ['DA', 'EB']]  # groups of related factors
c_grps = [distress, fear, externalizing + numb, externalizing]
grps = ['Mood', 'Hyper/Anxious Arousal', 'Anhedonia and Numbing', 'Externalizing']
res = [pd.concat([o.results_regressions_p[k] for k in cfas.results_regressions_p], axis=1) for o in [cfas, cfas_PTSD, cfas_n2]]
all_reg = pd.concat(res, keys=['Full', 'PTSD'])
sn = 1 if cfas.alpha >= 0.05 else 2 if cfas.alpha >= 0.01 else 3
for j, f in enumerate(f_grps):
    print('\n\n%s\n\n%s\n\n%s\n\n' % ('*' * 40, grps[j], '*' * 40))
    mat = all_reg.loc[:, :, f, :][c_grps[j]].applymap(lambda x: x if x.count('*') >= sn else '')
    if stack:
        mat = mat.stack().unstack(0).dropna(how='all').replace(np.nan, '')
    print(mat, '\n\n')


# %% Excel Results

# Write Results Tables to Excel
cfas.export_tables_excel(attributes=attrs_xl, sig='**')
cfas_n2.export_tables_excel(attributes=attrs_xl, sig='**')
cfas_PTSD.export_tables_excel(attributes=attrs_xl, sig='**')
bfs.export_tables_excel(attributes=attrs_xl, sig='**')

# Run R Analyses
if rerun_r:
    ro.r('source("analyses.R")')  # run R analyses script

# Check Data
# [obj.check_data() for obj in [cfas, cfas_PTSD, bfs]] # same NESARC-III & internal data?
# df = cfas.data.drop(cfas.mplus_index, axis=1).set_index('CASEID')
# comp = df == bfs.data.drop(bfs.mplus_index, axis=1).set_index('CASEID')[df.columns]
# print(comp.head()) # print results of True/False cfas-bfs data comparison
# print('\n\n%s\nBF-CFA Data Match:\n%s\n%s'%('='*80, str(comp.mean().describe()[['mean', 'min', 'max']]), '='*80))

# Check Python Attributes Against R & Excel Tables
cfas.check_tables_r()
cfas_PTSD.check_tables_r()
bfs.check_tables_r()
ans = dict(zip(['CFA', 'PTSD', 'BF'], [cfas, cfas_PTSD, bfs]))
ays = dict(zip(['Correlations', 'Regressions'], ['results_correlations', 'results_regressions']))
fig, axes = plt.subplots(nrows=len(ans), ncols=len(ays))
for a in ans:
    for i in ays:
        m, n = wh(list(ans.keys()), a), wh(list(ays.keys()), i)
        d = ans[a].check_results['difference'][ays[i]]
        axes[m, n].set_title('%s %s Python-R Difference' % (a, i))
        for y in range(len(d)):
            axes[m, n].hist(d[y], label=list(ans[a].DV_subsets_dictionary.keys())[y])
        axes[m, n].legend()
fig.tight_layout()
plt.get_current_fig_manager().window.showMaximized()
fig.show()
[i.max() for i in cfas_PTSD.check_results['difference']['results_regressions']]


# %% Open Documents

# Main
# os.system('chmod 744 %s'%cfas.documents['output']) # read/write permissions
# os.system('libreoffice --writer ' + cfas.documents['output']) # open files

# Supplement
# os.system('chmod 744 %s'%cfas.documents['output_supplement']) # read/write permissions
# os.system('libreoffice --writer ' + cfas.documents['output_supplement']) # open files


# %% Box Plots

# Correlation Box Plots
# fig, axes = plt.subplots(*square_grid(len(cfas.results_correlations)))
# rn_cats = {**dict(zip(['Dy', 'CM', 'Nu'], ['CM/Dy/Nu']*3)), 'In': 'In/Av', 'Av': 'In/Av'} # collapse some factors in plot
# rn_cats.update({'Dysphoric_Arousal': 'Dysphoric Arousal'}) # also re-name this model
# for by in ['Factor', 'Outcome']:
#     cfas.plot_boxplots(cats_by=by, exclude_vars='PTSD', rename_dict=rn_cats,
#                        k_depth='full', dodge=True, outlier_prop=cfas.alpha)
#     cfas_PTSD.plot_boxplots(cats_by=by, rename_dict=rn_cats, figsize=[26, 14], outlier_prop=cfas_PTSD.alpha)

fac_groups = [['In', 'Av'], ['CM', 'Dy', 'Nu', 'Ne', 'An'], ['HA', 'DA', 'AA', 'EB']]  # groups of related factors
fac_groups_keys = ['Intrusions/Avoidance', 'Mood', 'Arousal']  # labels to go with the groups
dv_groups = [['MDD', 'Depressed', 'Less_Accomplished'], ['GAD', 'Calm'], ['Phobia', 'Panic'],
             cfas.DV_subsets_dictionary['Drinking'] + ['Less_Careful'], ['AUD', 'NUD', 'BPD', 'CD', 'ASPD']]
dv_groups_keys = ['Depression (MDD, Depressed Feelings, and Less Accomplished)',
                  'Anxiety (GAD and Less Calm Feelings)',
                  'Fear (Specific Phobia and Panic Disorder)',
                  'Externalizing Behaviors (Drinking and Less Careful Behavior)',
                  'Externalizing Disorders (Personality and Substance Use Disorders)']
fac_order = ['In', 'Av', 'CM', 'Dy', 'Nu', 'An', 'Ne', 'HA', 'DA', 'AA', 'EB']
model_keys = ['Full Sample', 'PTSD Sub-Sample']
fig = cfas.plot_boxplots_compare(cfas_PTSD, attribute='results_correlations',
                                 cats_by='Factor', cats_order=fac_order,  # cats_by='Cluster',
                                 exclude_vars='PTSD', legend_cols=6, k_depth='full', dodge=True,
                                 figsize=[26, 14], fig_legend=True, outlier_prop=cfas.alpha, sharey=True,
                                 # ylim=[min_corr - 0.05, max_corr + 0.05],
                                 palette='bright', model_keys=model_keys, col_wrap=3, reverse_code='Calm',
                                 dv_groups=dv_groups, dv_groups_keys=dv_groups_keys, fac_groups=fac_groups,
                                 fac_groups_keys=fac_groups_keys, sharex=False)


# %% Heat Maps

# Results
types = ['results_correlations', 'results_regressions']
for o in models:
    sig = '*' if o.alpha >= 0.05 else '**' if o.alpha >= 0.01 else '***'
    for t in types:
        an = False if t == 'descriptives_factors' else True
        att = ['between', 'within'] if t == 'descriptives_factors' else [None]
        fig_loc = 'main' if t == 'results_correlations' else 'supplement'
        plot_args = {'figure_location': fig_loc, 'p_annotate': an, 'sig': sig, 'show': True}
        for a in att:
            o.plot_heatmaps(t, attribute_key=a, **plot_args)
    # o.plot_heatmaps_contrast(factor_contrasts, correlates, cbar_min_range=[-0.3, 0.3]) # hypotheses/contrasts

# Compare Analyses
cfas.compare_objects(cfas_PTSD, attributes=comp_attrs, object_names=['CFA', 'PTSD'], keep_dims=True,
                     clear_comparisons_attribute=clear_compare, plot=True)
cfas.compare_objects(bfs, attributes=comp_attrs, object_names=['CFA', 'Bifactor'], keep_dims=True,
                     clear_comparisons_attribute=False, plot=True)
for a in ['PTSD', 'Bifactor']:  # cycle through analyses comparisons
    hdr = ('=' * 80, '=' * 80, a, '=' * 80, '=' * 80)
    print('\n\n\n%s%s\n\n\t\t\t\t\tComparisons of CFA & %s Results\n\n%s%s\n\n' % hdr)
    res = cfas.results_compare_analyses[a]
    for r in comp_attrs:  # cycle through attributes compared
        print('\n%s\nSide by Side Comparison:\n%s \n\n\t\t\t' % ('*' * 40, '*' * 40), res[0][r])
        diff = [d if d.empty else d.describe().loc[['mean', 'min', 'max']] for d in res[1][r]]
        if all([d.empty for d in diff]):
            continue
        max_desc = [i.max() for i in [abs(d.loc[['min', 'max']]) for d in diff]]  # maximum difference for each DV
        max_y = dict(zip(cfas.DV_subsets_dictionary.keys(), [round(max(i), 3) for i in max_desc]))  # by subset
        print('\n%s\nDifferences Summary\n%s\n%s \n\n\t\t\t' % ('*' * 40, 'Maximum: %s' % str(max_y), '*' * 40), diff)

# Factor Correlations (Full & PTSD)
corr_mat = cfas.factor_scores.corr()
for r in range(corr_mat.shape[0]):
    for c in range(corr_mat.shape[1]):
        if (r == c) or (r < c):
            corr_mat.iloc[r, c] = cfas_PTSD.factor_scores.corr().iloc[r, c]  # put PTSD correlations on upper triangle
corr_mat = corr_mat.rename_axis(columns=['Model', 'Factor']).rename_axis(['Model', 'Factor'])
fig, axes = plt.subplots(figsize=(15, 10))
cmap = 'Reds' if np.mean(corr_mat >= 0).mean() == 1 else 'coolwarm'
tks = corr_mat.apply(
    lambda x: x.name[0] + '   ' + x.name[1] if x.name[1] == cfas.fac_series[x.name[0]][0] else x.name[1])
tks_top = corr_mat.apply(
    lambda x: x.name[0] + '\n' + x.name[1] if x.name[1] == cfas.fac_series[x.name[0]][0] else x.name[1])
tks, tks_top = [re.sub('_', ' ', t) for t in tks], [re.sub('_', ' ', t) for t in tks_top]
sb.heatmap(corr_mat, cmap=cmap, ax=axes, xticklabels=list(tks_top), yticklabels=list(tks))
axes.tick_params(top=True, bottom=False, left=True, labeltop=True, labelbottom=False)  # labels: top
plt.setp(axes.get_xticklabels(), rotation=-30, ha='right', rotation_mode='anchor')  # ticks
axes.tick_params(which='minor', bottom=False, left=False)
fig.tight_layout()
fig.savefig('Plots/factor_correlations_full_PTSD.jpeg')
cfas.factor_correlations_full_PTSD = corr_mat
cfas.files['figures']['main'].update({'factor_correlations_full_PTSD':
                                      ('Plots/factor_correlations_full_PTSD.jpeg', 'landscape')})
triangles_cap = 'Full Sample (Lower Triangle) and PTSD Sub-Sample (Upper Triangle)'
cfas.captions.update({'factor_correlations_full_PTSD': 'Factor Inter-Correlations for the %s' % triangles_cap})


# %% Tables & Figures to Word

# Set Up
cfas.initialize_documents(**doc_args)
cfas.define_styles(font_name='Times New Roman', font_size=12)

# Arguments
attrs_1 = ['models', 'indicators', 'descriptives_outcomes'] + attrs_wd[1:]
addl_attr = [['descriptives_outcomes'] + attrs_wd, attrs_wd]
orient = [dict(zip(attrs_1, ['portrait', 'portrait', 'landscape', 'portrait', 'landscape', 'landscape', 'landscape'])),
          dict(zip(addl_attr[0], ['landscape', 'portrait', 'portrait', 'landscape', 'landscape', 'landscape'])),
          dict(zip(addl_attr[1], ['portrait', 'portrait', 'landscape', 'landscape', 'landscape']))]
supp_tf = [False if x in ['models', 'fit'] else True for x in attrs_1]
types = ['fit', 'correlations', 'regressions']
types_supplement = [False, False, True]
addl_models = [cfas_PTSD, bfs]
models_supplement = [False, False, True]
addl_attributes = [addl_attr[0], addl_attr[0][1:]]
addl_supplement_tables = [True, True]

# Write Tables
# cfas.initialize_documents(**doc_args)
# cfas.export_tables_word(attrs_1, supplement_tables=supp_tf, write_results=True,
#                         addl_models=[cfas_PTSD, bfs], addl_supplement_tables=[True, True],
#                         addl_attributes=addl_attr, orient=orient)
# cfas.write_results(addl_models=addl_models,
#                    model_heads=['Overall Sample', 'PTSD Sub-Sample', 'Bifactor Models'],
#                    types=types, types_supplement=types_supplement,
#                    models_supplement=models_supplement)

# Write Just CFAs & PTSD Sub-Sample (No Bifactor)
cfas.export_tables_word(attrs_1, supplement_tables=supp_tf, write_results=True,
                        addl_models=addl_models, addl_supplement_tables=addl_supplement_tables,
                        addl_attributes=addl_attributes, orient=orient)
cfas.write_results(addl_models=[cfas_PTSD], model_heads=['Overall Sample', 'PTSD Sub-Sample'],
                   types=types, types_supplement=types_supplement[:2],
                   models_supplement=models_supplement[:2])

# Figures to Document(s)
cfas.export_figures_word(attributes=['results_correlations', 'factor_correlations_full_PTSD'],
                         parts=['main'], addl_models=None, reinitialize=True,
                         wide=[Inches(12), Inches(7)], tall=[Inches(7), Inches(12)],
                         note='Insignificant (p >= %s) effects blank.' % str(cfas.alpha))


# %% Syntax to Supplement


# %% Compose Manuscript

cfas.define_styles(font_name='Times New Roman', font_size=12)
cfas.compose_manuscript(['Manuscripts/Manuscript.docx', 'Manuscripts/Supplement/Supplement.docx'],
                        header_title='Validating Competing Structures')


# %% Commit

# Commit to GitHub
# message = 'Results from model run ' + str(datetime.now())
# # os.system('git add %'%str(os.getcwd() + '/Manuscripts/manuscript_PTSD_revised.docx'))
# # os.system('git add %'%str(os.getcwd() + '/Manuscripts/manuscript_PTSD_supplement_revised.docx'))
# os.system('git add *')
# os.system('git commit -m "%s\"'%message)
# os.system('git push')


# %% Manual Check Examples

tmp = pd.read_csv('MPlus/DSM_CFA_n2.csv')
cols="""B1
   B2
   B3
   B4
   B5
   C1
   C2
   D1
   D2
   D5
   D6
   D7
   E1
   E3
   E4
   E5
   E6
   A1
   W2S12Q15
   W2S12Q17
   W2S12Q25
   W2S12Q26
   W2S12Q27
   W2S12Q28
   W2S12Q32
   W2S12Q33
   ASPD
   PTSD_PY
   PTSD
   BPD
   PANIC_AL
   PANIC_AG
   PHOBIA
   GAD
   MDD
   AUD
   NUD
   SEX
   B3A
   B3B
   C2A
   C2B
   CALM
   DEPRESSE
   LESS_ACC
   LESS_CAR
   MAXDRINK
   USUALAMT
   DRINKFRE
   FREQMAX
   INTOX
   BINGE_MA
   BINGE_FE
   UPSET
   DISTRESS
   INTERFER
   HARDER_D
   RELATION
   PROBLEMS
   BINGE
   PANIC
   IN
   IN_SE
   AV
   AV_SE
   CM
   CM_SE
   HA
   HA_SE
   AUDWEIGH
   MPLUS_IN
   VARSTRAT
   VARUNIT"""
cols.split('\n')
[c.strip() for c in cols.split('\n')]
tmp = pd.read_csv('MPlus/DSM_CFA_n2.csv')
tmp.columns = [c.strip() for c in cols.split('\n')]
pars = []
for y in cfas_n2.DV_subsets_dictionary['SF-12']:
    exog = sm.add_constant(tmp[['IN', 'AV', 'CM', 'HA']])
    exog = sp.stats.zscore(exog, missing='drop')
    fff = sm.OLS(tmp[y[:min(len(y), 8)].upper()], exog).fit()
    fff.summary()
    pars += [fff.params.to_frame('Estimate').join(fff.pvalues.to_frame('P'))]
pars = pd.concat(pars, keys=cfas_n2.DV_subsets_dictionary['SF-12'])
print(pars.round(3))
