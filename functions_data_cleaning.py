#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=line-too-long
"""
Created on Mon Jan 11 12:59:24 2021

@author: ena
"""


# %% Imports

# Basics
# import rpy2.robjects.packages as rpackages
# import rpy2
import matplotlib.pyplot as plt
import seaborn as sb
import functools
import math
import re
import warnings
# import json
import pandas as pd
import numpy as np


def jsonKeys2int(x):
    """Convert keys. Modified: stackoverflow.com/questions/1450957/pythons-json-module-converts-int-dictionary-keys-to-strings."""
    nans = ['NaN', 'nan', 'np.nan', np.nan, '', '*']
    converts = [np.nan, np.nan, np.nan, np.nan, '', '*']
    if isinstance(x, dict):
        out = {wh(nans, k, converts) if k in nans else int(k): v for k, v in x.items()}
        return out
    return x


def jsonKeys2int_nested(x):
    """My expansion of the function I modified from SO."""
    return dict(zip(x.keys(), [jsonKeys2int(x[k]) for k in x]))


# Handy Variables for Printing
head = (str('\n\n\n' + '=' * 80 + '\n\n'), str('\n\n' + '=' * 80 + '\n\n'))


# %% Functions


def wh(x, y, z=0):
    """Search x list/vector for match with y; if z != 0, return element of z corresponding to location of match."""
    out = np.where([i == y for i in x])[0]
    out = [z[int(f)] for f in out] if z != 0 else out
    out = out[0] if len(out) == 1 else out
    return out


def square_grid(num_subplots):
    """Calculate number of rows & columns needed for subplots to be close to square."""
    rows = int(np.sqrt(num_subplots))  # number of rows (try to get close to square grid)
    cols = math.ceil(num_subplots / rows)  # number of columns
    return rows, cols


def try_float_format(x, digits=None, nsmall=None, warn=False,
                     ignore=None, recode=False, na_convert=['', 'nan', 'NaN', 'np.nan']):
    """Try to convert to float format."""
    try:
        for i in na_convert:
            x = re.sub(i, np.nan)
        if ignore is not None:
            x = re.sub(ignore, '', x)
        xf = x.astype(float) if type(x) in [pd.DataFrame, pd.Series] else float(x)
        if (nsmall is not None) and (digits is None):
            digits = nsmall
        if digits is not None:
            xf = round(xf, digits)
        if nsmall is not None:
            xf = xf.apply(lambda i: ('{:.%df}' % nsmall).format(i))
    except Exception:
        if warn:
            warnings.warn('Can\'t be floated')
        xf = x  # will return original
    return xf


def try_float(x):
    """Try to convert to float."""
    return try_float_format(x, digits=None, nsmall=None, warn=False, ignore=None, recode=False, na_convert='')


def dict_part(dictionary, part='items'):
    """Retrieve items or keys from a dictionary."""
    out = [dictionary[k] for k in dictionary] if part.lower() == 'items' else list(dictionary.keys())
    return out


def dict_rekey(dictionary, rename_dict=None, rc_flag=None):
    """Rekey dictionary based on a renaming dictionary."""
    if rename_dict is not None:  # if specify a rename dictionary...relabel keys if type_dict in original variable codes
        dictionary = dict([(rename_dict[k] if k in rename_dict.keys() else k, dictionary[k]) for k in dictionary])
    if rc_flag is not None:  # if specify a reverse-coding flag...change keys to remove suffix
        dictionary = dict([(re.sub(rc_flag, '', k), dictionary[k]) for k in dictionary])
    return dictionary


def type_convert(df, type_dict, rename_dict=None, rc_flag=None):
    """Convert types according to dictionary of types."""
    type_dict = dict_rekey(type_dict, rename_dict=rename_dict, rc_flag=rc_flag)  # re-key dictionary (if needed)
    ty_cs = wh([c in type_dict.keys() for c in df.columns], 1, list(df.columns))  # to convert type
    df.loc[:, ty_cs] = df[ty_cs].apply(lambda x: recode_cols_numeric(x, to_type=type_dict[x.name]))
    return df  # return dataframe with converted types


def get_multi_indicators(indicators, multiinds, data=None, threshold=1, na_flag=None):
    """Get counts of 1s (or values over threshold) to represent variables determined by multiple indicators."""
    if (na_flag is not None) and (data is not None):
        data = data.replace(na_flag, np.nan)
    inds = pd.Series(indicators, index=indicators)
    multi = multiinds if data is None else list(pd.Series([x if x in data.columns else np.nan for x in multiinds]).dropna())
    si = inds.apply(lambda x: wh([x in k for k in multi], 1, multi))  # series: lists of multiinds w/i indicators
    si = si.apply(lambda x: x if type(x) is list else [x])  # make sure all iterable, even if only 1 indicator
    if data is not None:
        counts = si.apply(lambda x: data[x].T.sum()).T
        if threshold is not None:
            counts = counts.applymap(lambda x: np.nan if pd.isnull(x) else 1 if x >= threshold else 0)
        counts = counts.drop(counts.columns.intersection(data.columns), axis=1)  # only retain new columns
        if na_flag is not None:
            counts = counts.replace(np.nan, na_flag)
        left_out = wh(si.apply(lambda x: len(x) == 0), 1, list(indicators))  # indicators for whom no multiinds in data
        left_out = left_out if type(left_out) is list else [left_out]  # make sure iterable
        data = pd.concat([data, counts], copy=False, axis=1)  # join with new variables
        if len(left_out) > 0:
            for x in left_out:
                data.loc[:, x] = np.nan  # set any indicators for whom no multiinds are available in data to NaN
        return si, data  # return counts (or dummy-coded, i.e. 1 for counts over threshold)
    else:
        return si, data  # just return series if not trying to convert data


def check_frequencies(df, check_dict, plot=True, rev_code_flag=None, na_flag='*'):
    """Check that frequencies in df match those expected as defined in check_dict."""
    keys = dict(zip(check_dict.keys(), [list(check_dict[k].keys()) for k in check_dict]))
    keys = dict(wh([k in df.columns for k in keys], True, [(k, keys[k]) for k in keys]))
    counts = pd.DataFrame([dict([(k, sum(df[c] == k)) for k in keys[c]]) for c in keys], index=keys).stack()  # counts
    chk = pd.concat([pd.Series(check_dict[k], index=check_dict[k].keys()) for k in check_dict], keys=check_dict.keys())
    check = pd.concat([chk, counts], axis=1).rename_axis(['Variable', 'Value'])
    check.columns = ['Expected', 'Observed']
    matches = check.apply(lambda x: x.Observed == x.Expected, axis=1).replace({True: 'Match', False: 'MISMATCH'})
    matches[check[pd.isnull(check.Expected)].index.values] = ''  # if no check for that variable-value, blank on match check
    check = check.assign(Match=matches).rename_axis(['Variable', 'Value'])
    if plot:  # plot observed & expected frequencies
        ch = check.drop('Match', axis=1).reset_index().replace(np.nan, -1).set_index(['Variable', 'Value']).stack(0)
        ch = ch.rename_axis(['Variable', 'Value', 'Source']).reset_index([1, 2]).rename({0: 'Count'}, axis=1)
        fig, axes = plt.subplots()
        for v in range(len(keys)):
            fig.add_subplot(*square_grid(len(keys)), v + 1)
            d, var = ch.loc[list(keys.keys())[v]], list(keys.keys())[v]
            d = d.replace({'Value': dict(zip(keys[var], np.arange(len(keys[var]))))})
            sb.barplot(data=d.reset_index(), x='Value', y='Count', hue='Source')
            plt.gca().set_title(var)
            plt.gca().set_xticks(keys[var])
        fig.legend()
    misses = check[check.Match == 'MISMATCH']
    if len(misses.index.values) > 0:
        print(misses)
        warnings.warn('\n\n\n%s\n\nMISMATCHES FOUND!\n\n%s\n' % ('=' * 80, '=' * 80))
    else:
        print(check, '\n\n\n%s\n\nAll looks good!\n\n%s\n' % ('=' * 80, '=' * 80))
    df_cols = list(df.columns)
    if rev_code_flag is not None:
        df_cols = wh([rev_code_flag in c for c in df_cols], 0, df_cols)  # to not warn a/b r-c
    no_check = [x + ' ' if x not in check_dict.keys() else '' for x in df_cols]  # variables not in check dictionary
    no_check = re.sub(' *$', '', functools.reduce(lambda i, j: i + j, no_check))
    if len(no_check) > 0:
        print('\n\n%s\n\nVariables in dataframe but not in checks:\n\t%s\n\n%s' % ('=' * 100, no_check, '=' * 100))
    try:
        tm = pd.concat([df[s].value_counts(sort=False).T.loc[keys[s]] for s in keys],
                       keys=check_dict.keys()).unstack(0)
        tx = pd.concat([pd.DataFrame(check_dict[k], index=[k], columns=tm[k].index.values) for k in check_dict]).T
        if tx.compare(tm).empty:
            print('')
            print('\n\n\n%s\n\nFrequencies checked out via alternate method! Hooray!\n\n%s\n' % ('*' * 80, '*' * 80))
        else:
            print('\n\n\n%s\n\nMISMATCHES DETECTED USING ALTERNATE METHOD\n\n%s\n' % ('*' * 80, '*' * 80))
    except Exception as err:
        print(err, '\n\n\n%s\n\nFAILED TO RE-CHECK VIA ALTERNATE METHOD\n\n%s\n' % ('*' * 80, '*' * 80))
    return check, check[check.Match == 'MISMATCH']  # check & mismatched subset


def reverse_code_data(df, reverse_code_dictionary, suffix='_r'):
    """Reverse code data according to a dictionary."""
    old_names = list(reverse_code_dictionary.keys())
    new_names = [re.sub(suffix, '', x) for x in old_names]
    data = df.replace(reverse_code_dictionary)
    data = data.rename(dict(zip(old_names, new_names)), axis=1)
    return df.join(data[new_names], lsuffix='_original')


def reverse_code(values, data=None, shift_to_zero=False, na_flags=None, first_replace=None):
    """Reverse-code values."""
    if type(na_flags) not in [np.array, list, type(None)]:
        na_flags = [na_flags]  # make iterable if length 1
    arr = list(np.arange(1, values + 1)) if type(values) in [int, float] else values
    arr = [int(a) for a in arr]
    reversed_values = arr.copy()
    reversed_values.reverse()  # happens in-place
    if shift_to_zero:
        reversed_values = [i - min(reversed_values) for i in reversed_values]  # make start at zero if wanted
    replace_dict = dict(zip(arr, [reversed_values[v] for v in range(len(reversed_values))]))
    replace = replace_dict if na_flags is None else {**replace_dict, **dict(zip(na_flags, [np.nan] * len(na_flags)))}
    if data is not None:
        if first_replace is not None:
            data = data.replace(first_replace)
        out = data.replace({**replace, **dict(zip(na_flags, [np.nan] * len(na_flags)))})
    else:
        out = replace
    return out


def recode_strings(data, str_cols=None, na_flag='*'):
    """Recode strings as numeric."""
    if str_cols is None:  # find string columns (after replacing na_flags) if not specified
        str_cols = data.columns[wh([type(data.replace(na_flag, np.nan)[x][0]) == str for x in data.columns], 1)]
    if type(str_cols) not in [list, type(None)]:
        str_cols = [] if str_cols is None else [str_cols]  # make sure iterable
    if len(str_cols) > 0:
        print('Re-coding string variables.\n', str_cols)
        print('\n\n\nBEFORE: \n\n', data[str_cols].head())
        if (len(str_cols) == 1) and (len(pd.unique(data[str_cols[0]])) == data.shape[0]):  # if unique strings & 1 variable
            old_vars, new_vars = data.loc[:, str_cols[0]], np.arange(data.shape[0])  # faster to do it this way
            data.loc[:, str_cols[0]] = new_vars
            recode_map = {str_cols[0]: dict(zip(new_vars, old_vars))}
        else:
            rmp = data[str_cols].apply(lambda x: dict(zip(np.arange(len(pd.unique(x))), np.array(pd.unique(x)))))
            replace_map = dict(zip(str_cols, [dict(zip(dict_part(r, 'items'), list(r.keys()))) for r in rmp]))
            recode_map = dict(zip(str_cols, rmp))  # dictionary to convert back to original
            data = data.replace(replace_map)  # replace with numeric re-coding
            data.loc[:, str_cols] = data.loc[:, str_cols].astype(int)  # make sure integers
        print('\n\n\nAFTER: \n\n', data[str_cols].head())
    return data, recode_map


def recode_cols_numeric(x, to_type='float', na_flag='*'):
    """Convert variable types to to_type."""
    if to_type in ['str', 'int', 'integer']:
        return x if to_type == 'str' else x.apply(lambda i: i if i in [na_flag, np.nan] else int(try_float(i)))
    else:
        return pd.to_numeric(x.replace(na_flag, np.nan), errors='ignore', downcast=to_type).replace(np.nan, na_flag)


def print_reverse_dict(dictionary, obj_name='obj'):
    """Print a reversed dictionary."""
    tab_length = len('%s = {' % obj_name)
    tmp = ['\'%s\': \'%s\'' % (i[0], i[1])for i in zip(dict_part(dictionary, 'items'), list(dictionary.keys()))]
    dict_str = functools.reduce(lambda x, y: '%s,\n%s%s' % (x, ' ' * tab_length, y), tmp)
    print('%s = {' % obj_name + dict_str + '}')


def reverse_dict(dictionary):
    """Return a dictionary with the previous items as keys, and the previous keys as items."""
    return dict(zip([dictionary[k] for k in dictionary], list(dictionary.keys())))


# %% Create Unit Test Data for Reverse-Code Checks

scale_tops, na_flags = [5, 11], [[9], [99, 999]]  # top of scales, NaN flags for 'A' & 'B' (& 'A' variations 'C', 'D', & 'E')
scale_type_vars = [['A', 'C', 'D', 'E'], ['B', 'F']]
expect_dicts = {1: 5, 2: 4, 3: 3, 4: 2, 5: 1}, dict(zip(range(1, 12), [np.arange(1, 12)[-i] for i in range(1, 12)]))
expect_dicts_full = [{**expect_dicts[x], '': 0, **[{9: np.nan}, {99: np.nan, 999: np.nan}][x]} for x in range(2)]
blanks, nas = [[4, 2, 20], [1, 6, 4, 12]], [[7, 11], [20, 3, 5]]  # different index locations for modifications
data = pd.DataFrame({'A': ['', 3, 4, 1, 5, '', 1, 9, 4, 5, 5, 4, 2, 1, 4, 1, 4, 4, 4, 2, 5, 4, 2, 3, 9],
                     'B': ['', 11, 5, 10, 10, 6, 2, 1, 10, 9, 6, 99, 3, 3, 9, 6, 5, 7, 8, 999, 99, 3, 7, 11, ''],
                     'C': [1, 3, 4, 1, 5, 1, 1, 5, 4, 5, 5, 4, 2, 1, 4, 1, 4, 4, 4, 2, 5, 4, 2, 3, 2],
                     'D': ['', 3, 4, 1, 5, '', 1, 5, 4, 5, 5, 4, 2, 1, 4, 1, 4, 4, 4, 2, 5, 4, 2, 3, 2],
                     'E': [1, 3, 4, 1, 5, 1, 1, 9, 4, 5, 5, 4, 2, 1, 4, 1, 4, 4, 4, 2, 5, 4, 2, 3, 9],
                     'F': ['', 11, 5, 10, 10, 6, 2, 1, 10, 9, 6, 999, 3, 3, 9, 6, 5, 7, 8, 1, 99, 3, 7, 11, '']})
exp = pd.DataFrame({'A': [0, 3, 2, 5, 1, 0, 5, np.nan, 2, 1, 1, 2, 4, 5, 2, 5, 2, 2, 2, 4, 1, 2, 4, 3, np.nan],
                    'B': [0, 1, 7, 2, 2, 6, 10, 11, 2, 3, 6, np.nan, 9, 9, 3, 6, 7, 5, 4, np.nan, np.nan, 9, 5, 1, 0],
                    'C': [5, 3, 2, 5, 1, 5, 5, 1, 2, 1, 1, 2, 4, 5, 2, 5, 2, 2, 2, 4, 1, 2, 4, 3, 4],
                    'D': [0, 3, 2, 5, 1, 0, 5, 1, 2, 1, 1, 2, 4, 5, 2, 5, 2, 2, 2, 4, 1, 2, 4, 3, 4],
                    'E': [5, 3, 2, 5, 1, 5, 5, np.nan, 2, 1, 1, 2, 4, 5, 2, 5, 2, 2, 2, 4, 1, 2, 4, 3, np.nan],
                    'F': [0, 1, 7, 2, 2, 6, 10, 11, 2, 3, 6, np.nan, 9, 9, 3, 6, 7, 5, 4, 11, np.nan, 9, 5, 1, 0]})


# %% Check Process

replace = [reverse_code(scale_tops[x]) for x in range(2)]
if all([replace[x] == expect_dicts[x] for x in range(2)]):
    print('Successful check of reverse-code only replacement dictionary match.')
else:
    raise Exception('Failed check of reverse-code only replacement dictionary match.')

df = data.copy()
for x in range(len(scale_type_vars)):
    locs = scale_type_vars[x]
    df.loc[:, locs] = reverse_code(scale_tops[x], first_replace={'': 0}, na_flags=na_flags[x], data=df[locs])
print(df.compare(exp, keep_shape=True).replace({np.nan: 'Same'}))
if df.compare(exp).empty:
    print('Successful check of reverse-coding with replacement in dataframe.')
else:
    raise Exception('Failed check of reverse-coding with replacement in dataframe.')
