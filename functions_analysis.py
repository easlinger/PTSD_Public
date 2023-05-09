#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=line-too-long
"""
Created on Mon Apr 19 15:24:30 2021.

@author: ena
"""

import warnings
import scipy as sp
import statsmodels.api as sm
import pandas as pd
import numpy as np
from functions_data_cleaning import wh, dict_part, try_float_format


def pearsonr_ci(x_var, y_var, alpha=0.01):
    """
    Calculate Pearson correlation along with the confidence interval using scipy and numpy.

    Parameters
    ----------
    x_var, y_var : iterable object such as a list or np.array
      Input for correlation calculation
    alpha : float
      Significance level. 0.05 by default
    Returns
    -------
    r : float
      Pearson's correlation coefficient
    pval : float
      The corresponding p value
    lo, hi : float
      The lower and upper bound of confidence intervals

    Notes
    -----
    From https://zhiyzuo.github.io/Pearson-Correlation-CI-in-Python/)

    """
    cor, pVal = sp.stats.pearsonr(x_var, y_var)
    r_z = np.arctanh(cor)
    s_e = 1 / np.sqrt(x_var.size - 3)
    zVal = sp.stats.norm.ppf(1 - alpha / 2)
    lo_z, hi_z = r_z - zVal * s_e, r_z + zVal * s_e
    low, high = np.tanh((lo_z, hi_z))
    return cor, pVal, [low, high]


def p_st(value, thresholds):
    """Return *, **, ***, or nothing depending on significance."""
    stars = ''  # start with no stars
    for t in sorted(thresholds, reverse=True):  # go in reverse order
        if np.float128(value) < np.float128(t):
            stars = stars + '*'
    return stars


def test_p_stars(p_th):
    """Test that p_stars function behaves as expected."""
    tests = [p_st(p, p_th) for p in np.random.uniform(p_th[0], 1, size=20)]
    if len(p_th) >= 1:
        tests = tests + [p_st(p, p_th) for p in np.random.uniform(p_th[1], p_th[0], size=20)]
    if len(p_th) >= 2:
        tests = tests + [p_st(p, p_th) for p in np.random.uniform(p_th[2], p_th[1], size=20)]
    if len(p_th) >= 3:
        tests = tests + [p_st(p, p_th) for p in np.random.uniform(0, p_th[2], size=20)]
    mess = ['p > %d' % p_th[0]] + ['p < %s' % str(i) for i in p_th]
    for t in range(len(p_th) + 1):
        if any([i != '*' * t for i in tests[t]]):
            raise Exception('Unexpected output for test of threshold for %s' % mess[t])


def p_stars(value, thresholds=None):
    """Use wrapper for p-starring function & test function."""
    if thresholds is None:
        thresholds = [0.05, 0.01, 0.001]
    test_p_stars(thresholds)  # make sure expected behavior
    stars = p_st(value, thresholds)
    return stars


def stars_only(data):
    """Remove numbers from coefficient columns with p-value stars."""
    return data.apply(lambda x: x.astype(str).str.strip('-|0|1|2|3|4|5|6|7|8|9.'))


def corr_dfs(df_1, df_2=None, digits=3, alpha=0.01):
    """Correlations of columns in 1 or 2 dataframes (NAs dropped)."""
    if df_2 is None:
        df_2 = df_1.copy()  # if 2nd dataframe is not specified, copy 1st
    corr_mat = pd.DataFrame(np.nan, columns=df_2.columns, index=df_1.columns)
    corr_mat_p = pd.DataFrame(columns=df_2.columns, index=df_1.columns)
    corr_mat_t = pd.DataFrame(columns=df_2.columns, index=df_1.columns)
    if (df_1.dropna().empty is False) and (df_2.dropna().empty is False):
        for r in df_1.columns:
            for c in df_2.columns:
                if (r not in df_1.columns) or (c not in df_2.columns):
                    continue
                x_x = df_1[r] if df_1[[r]].shape[1] == 1 else df_1[r].iloc[:, 0]
                y_y = df_2[c] if df_2[[c]].shape[1] == 1 else df_2[c].iloc[:, 0]
                drop = np.logical_or(np.array((pd.isnull(x_x))), np.array(pd.isnull(y_y)))  # if NA in either
                if (df_1[~drop][[r]].dropna().empty) or (df_2[~drop][[c]].dropna().empty):
                    continue  # if r or c all NaN
                x_x, y_y = x_x[~drop], y_y[~drop]
                coef, pval, bounds = pearsonr_ci(x_x, y_y, alpha=alpha)  # correlation & p-value
                corr_mat.loc[r, c] = coef  # store correlation coefficient
                corr_mat_p.loc[r, c] = ('{:.%df}' % digits).format(coef) + p_stars(pval)  # with significance stars
                corr_mat_t.loc[r, c] = '%s [%s, %s]' % tuple([('{:.%df}' % digits).format(x) for x in [coef] + bounds])
    corr_mat_t.columns = [f'{c} ({100 * (1 - alpha)}% CI)' for c in df_2.columns]
    return corr_mat, corr_mat_p, corr_mat_t


def corr(x_x, y_y, digits=3):
    """Find correlation with significance."""
    rs = y_y.join(x_x).corr(method=lambda x, y: sp.stats.pearsonr(x_x, y_y)[0]).loc[x_x.columns][y_y.columns]
    ps = y_y.join(x_x).corr(method=lambda x, y: sp.stats.pearsonr(x_x, y_y)[1]).loc[x_x.columns][y_y.columns]
    rs2 = rs.applymap(lambda x: ('{:.%df}' % digits).format(x_x))  # rounding & precision
    ps2 = rs2 + ps.applymap(lambda x: ''.join(p_stars(x_x)))  # with significance stars
    return rs, ps2  # return correlations & correlations with stars


def regress(yy, x, log=False, na_p='drop', digits=2, alpha=0.01, multiple_y=True, stdx=True):
    """Perform regressions for multiple outcome variables & return formatted output of various types."""
    ci_cols = ['[%s' % '{:.3f}'.format(alpha / 2), '%s]' % '{:.3f}'.format(1 - alpha / 2)]  # confidence interval columns
    # ix, cols = x.index, x.columns if log else x.index, ['b0'] + list(x.columns)
    ix, cols = x.index, ['b0'] + list(x.columns) if log is False else list(x.columns)
    if stdx:
        x = sp.stats.zscore(x, nan_policy='omit')  # standardize predictors (if desired)
    if log is False:
        x = sm.add_constant(x, 1)
    # x = sm.add_constant(x, 1)
    x = pd.DataFrame(x, index=ix)
    x.columns = cols
    params = pd.DataFrame(index=cols, columns=yy.columns)
    params_p = pd.DataFrame(index=cols, columns=yy.columns)
    rsquared = pd.DataFrame(columns=cols, index=['R-Squared', 'F-Statistic'])
    fitted, results = dict(), dict()  # empty dictionaries for results & fit attribute
    tables = []  # empty list for summary tables
    for i in yy.columns:
        try:
            y = yy[i]
            res = sm.Logit(y, x, missing=na_p) if log else sm.OLS(y, x, missing=na_p)  # results
            fit = res.fit()  # fit attribute of results
            tab = fit.summary2(alpha=alpha).tables[1]
            tab = tab.assign(r=[fit.prsquared if log else fit.rsquared] + [''] * (len(fit.params) - 1))  # r^2
            rsq_name = ['Pseudo-R-Squared' if log else 'R-Squared'][0]  # R^2
            tab = tab.rename({'Std.Err.': 'SE', 'Coef.': 'Estimate', 'r': rsq_name,  # re-name summary columns
                              wh(['P>' in c for c in tab.columns], True, list(tab.columns)): 'P-Value'}, axis=1)
            tab = tab[['Estimate', 'P-Value', '[%s' % str(alpha / 2), '%s]' % str(1 - alpha / 2), rsq_name]]
            if log:  # -> log odds if logistic
                tab.loc[:, 'Estimate'] = np.exp(tab.loc[:, 'Estimate'])
                tab.loc[:, ci_cols[0]] = np.exp(tab.loc[:, ci_cols[0]])
                tab.loc[:, ci_cols[1]] = np.exp(tab.loc[:, ci_cols[1]])
            tab = tab.set_index(pd.MultiIndex.from_tuples([(i, c) for c in tab.index]))
            tab.loc[:, 'P-Value'] = ['{:.4f}'.format(float(p)) for p in tab.loc[:, 'P-Value']]
            pvals = fit.pvalues  # p-values to be turned into stars below
            sig = [p_stars(p) for p in pvals]
            par = np.exp(fit.params) if log else fit.params  # log odds or coefficients
            # par = [str(('{:.%df}'%digits).format(b)) for b in par] # coefficients
            par_p = [str(('{:.%df}' % digits).format(b)) + str(p) for b, p in zip(par, sig)]  # with stars
            params.loc[:, i] = par
            params_p.loc[:, i] = par_p
            rsq = fit.prsquared if log else fit.rsquared  # r-squared
            # rsquared.loc['R-Squared', i] = str('{:.%df}'%digits).format(rsq) # format r-squared
            rsquared.loc['R-Squared', i] = rsq  # r-squared
            # rsquared.loc['F-Statistic', i] = str('{:.%df}'%digits).format(F) # format r-squared
            tables = tables + [tab]  # add sorted summary table to list
            fitted.update({i: fit})
            results.update({i: res})
        except Exception as e:
            print(e, 'Could not complete regression function for outcome ' + i)
    tables = pd.concat(tables)
    cis = tables[['Estimate'] + ci_cols].applymap(lambda i: ('{:.%df}' % digits).format(i))  # format estimates & CIs
    cis = cis.apply(lambda j: '%s [%s, %s]' % tuple(j[['Estimate'] + ci_cols]), axis=1)  # paste: Estimate [CI_LB-CI_UB]
    return fitted, rsquared, params, params_p, results, tables, cis


def try_regress(yy, x, log=False, alpha=0.01, **kwargs):
    """Try the regress function."""
    try:
        return regress(yy, x, log=log, alpha=alpha, **kwargs)
    except Exception as err:
        print(f'{err}\n\nError in try_regress.')
        cols = ['b0'] + list(x.columns) if log is False else list(x.columns)
        ci_cols = ['[%s' % '{:.3f}'.format(alpha / 2), '%s]' % '{:.3f}'.format(1 - alpha / 2)]  # confidence interval columns
        rsq_name = ['Pseudo-R-Squared' if log else 'R-Squared'][0]  # R^2
        tab = pd.DataFrame(index=pd.MultiIndex.from_product([yy.columns, cols]),
                           columns=['Estimate', 'P-Value', '[%s' % str(alpha / 2), '%s]' % str(1 - alpha / 2), rsq_name])
        cis = tab[['Estimate'] + ci_cols]
        params = pd.DataFrame(index=cols, columns=yy.columns)
        params_p = pd.DataFrame(index=cols, columns=yy.columns)
        res = dict(zip(yy.columns, [] * len(yy.columns)))
        rsq = pd.DataFrame(index=['R-Squared', 'F-Statistic'], columns=['b0'] + cols)
        return [res, rsq, params, params_p, res, tab, cis]


def compare_data(data_a, data_b, data_list=True, align=0, keep_dims=True, keep_same=False, object_names=None):
    """Compare dataframes."""
    if object_names is None:
        object_names = ['self', 'other']
    data_a, data_b = [dict_part(d, 'items') if type(d) == dict else d for d in [data_a, data_b]]  # list if dictionary
    if data_list is False:
        data_a, data_b = [data_a, data_b]  # make iterable if only one df in each
    ix_intersect = [data_a[d].index.intersection(data_b[d].index) for d in range(len(data_a))]  # indices intersection
    col_intersect = [data_a[d].columns.intersection(data_b[d].columns) for d in range(len(data_a))]  # column intersect
    da, db = [[x[d].loc[ix_intersect[d], col_intersect[d]] for d in range(len(x))] for x in [data_a, data_b]]
    da, db = [[try_float_format(d) for d in x] for x in [da, db]]  # try to convert to float
    compare = [np.nan] * len(da)
    diff = [np.nan] * len(da)
    contr = [np.nan] * len(da)
    for d, q in enumerate(da):
        try:
            comp = d.compare(db[q], keep_shape=keep_dims, keep_equal=keep_same, align_axis=align)  # df compare
        except Exception as err:
            print(err)
            warnings.warn('\n\n Could not compare on element %d' % q)
            continue
        try:
            comp = comp.reset_index(-1).replace('self', object_names[0]).replace('other', object_names[1])
            comp = comp.reset_index().set_index(list(comp.reset_index().columns[:(len(comp.index.names) + 1)]))
            comp = comp.rename_axis([None] * len(comp.index.names))  # remove row index labels
        except Exception as name_err:
            print(name_err)
            warnings.warn('\n\n Could not rename index after objects on element %d' % d)
            continue
        comp = comp.applymap(try_float_format)  # try to convert comp cells to float
        comp_num = comp.select_dtypes('number')  # select numeric columns
        comp_str = comp.drop(list(comp_num.columns), axis=1)  # select non-numeric columns
        ca_n, cb_n = comp_num.loc[:, :, object_names[0]], comp_num.loc[:, :, object_names[1]]  # self & other (numeric)
        ca_s, cb_s = comp_str.loc[:, :, object_names[0]], comp_str.loc[:, :, object_names[1]]  # """ (non-numeric)
        try:
            diff[d] = ca_n.subtract(cb_n)  # difference between dfs
        except Exception as numeric_err:
            print(numeric_err)
            warnings.warn('\n\n Numeric comparison failed on element %d' % d)
        try:
            contr[d] = ca_s.compare(cb_s, keep_equal=False, keep_shape=True).replace(np.nan, '')  # non-numeric compare
        except Exception as str_err:
            print(str_err)
            warnings.warn('\n\n String comparison failed on element %d' % d)
        compare[d] = comp
    return [compare, diff, contr]


def descriptives_tables(df, DV_subsets_dictionary, group=None, digits=2, group_levels=None):
    """Descriptive statistics tables."""
    ydi = DV_subsets_dictionary if DV_subsets_dictionary is not None else df.columns
    fst = str('{:.%df}' % digits)  # formatting string for rounding & precision (digits = minimum digits after decimal)
    cat_i = [np.array([len(df[i].unique()) == 2 for i in ydi[k]]).all() for k in dict_part(ydi, 'keys')]
    cat = wh(cat_i, True, dict_part(ydi, 'keys'))  # use index created above to ID DV subsets that need N & %
    if sum(cat_i) == 1:
        cat = list([cat])  # turn into list if only 1 subset is categorical
    cont = wh(cat_i, False, dict_part(ydi, 'keys'))
    if sum([c is False for c in cat_i]) == 1:
        cont = list([cont])  # turn into list if only 1 subset is continuous
    group_cols = ['Overall'] if group is None else ['Overall'] + list(df[group].unique())  # group headers
    descriptives_categorical, descriptives_continuous = [np.nan, np.nan]  # empty to start
    if len(cat) > 0:  # if any categorical subsets
        if sum(cat_i) == 1:  # if just 1 categorical subset, make sure not nested list
            row_i_cat = pd.MultiIndex.from_tuples([[(k, i) for i in ydi[k]] for k in cat][0])
        else:
            exploded_i_cat = list(pd.Series([[(k, i) for i in ydi[k]] for k in cat]).explode())  # tuples all in 1
            row_i_cat = pd.MultiIndex.from_tuples(exploded_i_cat)  # multi-index (rows)
        c_i_cat = pd.MultiIndex.from_tuples([(k, i) for k in group_cols for i in ['Count', 'Percent']])  # index
        cat_table = pd.DataFrame(index=row_i_cat, columns=c_i_cat)  # table from multi-index
        y_cat = list(pd.Series([[i for i in ydi[k]] for k in cat]).explode())  # categorical variables
        ns = cat_table.apply(lambda a: '{:.0f}'.format(df[a.name[1]].dropna().sum()), axis=1)  # overall ns
        cat_table.loc[:, pd.IndexSlice['Overall', 'Count']] = ns  # assign overall counts
        percs = cat_table.apply(lambda a: fst.format(100 * df[a.name[1]].dropna().mean()), axis=1)  # overall %
        cat_table.loc[:, pd.IndexSlice['Overall', 'Percent']] = percs  # assign overall %s
        if group is not None:  # if any groups
            for g in group_cols[1:]:  # iterate through groups & retrieve counts
                c_g = [fst.format(sum(df[df[group] == g][y].dropna())) for y in y_cat]
                p_g = [fst.format(100 * df[df[group] == g][y].dropna().mean()) for y in y_cat]
                cat_table.loc[:, pd.IndexSlice[g, 'Count']] = c_g  # assign counts for group
                cat_table.loc[:, pd.IndexSlice[g, 'Percent']] = p_g  # assign counts for group
            if group_levels is not None:
                cat_table = cat_table.rename(dict(zip(np.arange(len(group_levels)), group_levels)), axis=1)
        else:
            cat_table = cat_table.stack(0).reset_index(2, drop=True)  # if no groups, remove "Overall" header
        if len(cat) == 1:
            cat_table = cat_table.reset_index(0, drop=True)  # drop subset name index if only one
        descriptives_categorical = cat_table  # assign as attribute
    if len(cont) > 0:  # if any continuous subsets
        if sum([c is False for c in cat_i]) == 1:  # if just 1 continuous subset, make sure not nested list
            row_i_cont = pd.MultiIndex.from_tuples([[(k, i) for i in ydi[k]] for k in cont][0])
        else:  # otherwise, put tuples all in same list
            exploded_i_cont = list(pd.Series([[(k, i) for i in ydi[k]] for k in cont]).explode())  # tuples in 1
            row_i_cont = pd.MultiIndex.from_tuples(exploded_i_cont)
        c_i_cont = pd.MultiIndex.from_tuples([(k, i) for k in group_cols for i in ['Mean', 'SD']])  # multi-index
        cont_table = pd.DataFrame(index=row_i_cont, columns=c_i_cont)  # table from multi-index
        y_cont = list(pd.Series([[i for i in ydi[k]] for k in cont]).explode())  # continuous variables
        mus = cont_table.apply(lambda a: fst.format(df[a.name[1]].dropna().mean()), axis=1)  # overall means
        cont_table.loc[:, pd.IndexSlice['Overall', 'Mean']] = mus  # assign overall means
        sds = cont_table.apply(lambda a: fst.format(df[a.name[1]].dropna().std()), axis=1)  # overall SDs
        cont_table.loc[:, pd.IndexSlice['Overall', 'SD']] = sds  # assign overall SDs
        if group is not None:  # if any groups
            for g in group_cols[1:]:  # iterate through groups & retrieve counts
                m_g = [fst.format(df[df[group] == g][y].dropna().mean()) for y in y_cont]
                s_g = [fst.format(df[df[group] == g][y].dropna().std()) for y in y_cont]
                cont_table.loc[:, pd.IndexSlice[g, 'Mean']] = m_g  # assign means for group
                cont_table.loc[:, pd.IndexSlice[g, 'SD']] = s_g  # assign SDs for group
            if group_levels is not None:
                cont_table = cont_table.rename(dict(zip(np.arange(len(group_levels)), group_levels)), axis=1)
        else:
            cont_table = cont_table.stack(0).reset_index(2, drop=True)  # if no groups, remove "Overall" header
        if len(cont) == 1:
            cont_table = cont_table.reset_index(0, drop=True)  # drop subset name index if only one
        descriptives_continuous = cont_table  # assign as attribute
    return descriptives_categorical, descriptives_continuous
