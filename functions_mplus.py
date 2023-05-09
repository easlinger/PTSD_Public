#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=line-too-long
"""
Created on Wed Dec  9 11:13:30 2020.

@author: ena
"""

import os
import functools
import numpy as np
import pandas as pd
import scipy as sp
import re
import warnings
import rpy2
from rpy2.robjects import pandas2ri
import rpy2.robjects.packages as rpackages
from functions_data_cleaning import wh
# import rpy2.robjects as ro
# From https://github.com/ejolly
os.environ[
    "KMP_DUPLICATE_LIB_OK"
] = "True"  # Recent versions of rpy2 sometimes cause the python kernel to die when running R code; this handles that
R_LIBS = rpy2.robjects.r('.libPaths()')[0]
base = rpackages.importr('base')
utils = rpackages.importr('utils')
stats = rpackages.importr('stats')
mplus = rpackages.importr('MplusAutomation', lib_loc=R_LIBS)
openxlsx = rpackages.importr('openxlsx', lib_loc=R_LIBS)


def add_to_string_wrapped(additions, string='', end='', limit=60):
    """Add strings list, wrapping at some character limit."""
    length = len(string)  # starting length
    for x in additions:  # add other strings, breaking lines when too many characters
        same_line, next_line = '%s %s' % (string, x), '%s\n\t\t%s' % (string, x)
        string, length = [same_line, length + len(x) + 1] if length + len(x) < limit else [next_line, len('\t%s' % x)]
    return string + end


def cfa_syntax(model, factors, indicators, data_column_names, data_file='n3.csv', out_file=0,
               faster=False, free_load=True, standardization='standardized', categorical=None, na_flag='*',
               preview=True, plots=True, cores=None, threads=None, starts=None, stit=None,
               cluster=None, stratification=None, weight=None, write=True, id_variable=None,
               analysis='CFA', subpopulation=None, ESTIMATOR='wlsmv', TYPE=None, auxiliary=True,
               variable_statement_extras=None, analysis_statement_extras=None, model_statement_extras=None,
               output_statement_extras=None, plot_statement_extras=None, save_statement_extras=None,
               bayes_fscores_imputations=[50, 10], montecarlo=None, bayes_biterations=20000, interactive=None):
    """Write Mplus input files/syntax for factor analyses."""
    # Set Up
    out_file = model + '_' + analysis if out_file == 0 else out_file  # default .inp/.out & .csv file stem
    f_ser = pd.Series(indicators, index=factors)  # series with indicators by factor
    col_names = add_to_string_wrapped(data_column_names)  # paste & wrap use variables at character limit
    if categorical is not None:
        if categorical:
            categorical = list(np.array(f_ser.explode()))  # categorical = all indicators if True
        categ_vars = add_to_string_wrapped(categorical)  # paste & wrap categorical variables at character limit

    # Remove Incompatible Options
    if ESTIMATOR.lower() == 'bayes':
        if TYPE.lower() == 'complex':
            warnings.warn('TYPE = COMPLEX cannot be used with BAYES estimator. Removing specification.')
            TYPE = None

    # Data & Variable Statements
    d_stmnt = '\nDATA: FILE IS \'%s\';' % data_file
    use_names = add_to_string_wrapped(list(np.array(f_ser.explode())))  # paste & wrap use variables at character limit
    idvar = '\n\tIDVARIABLE = %s;' % id_variable if id_variable is not None else ''  # id column
    v_stmnt = '\nVARIABLE:\n\tNAMES =%s;\n\tUSEVARIABLES =%s; %s' % (col_names, use_names, idvar)
    v_args = [cluster, stratification, weight, subpopulation]  # possible extra variable statements
    v_extras = dict(zip(['cluster', 'stratification', 'weight', 'subpopulation'], v_args))
    v_extras_nones = wh([x is None for x in v_args], True, list(v_extras.keys()))  # which not none
    if len(v_extras_nones) > 0:
        if type(v_extras_nones) is not list:
            v_extras_nones = [v_extras_nones]  # ensure iterable
        for n in v_extras_nones:  # remove unspecified extra variable options
            v_extras.pop(n)
    ve = [i + ' = ' + v_extras[i] + ';' for i in v_extras]
    if auxiliary:  # auxiliary variables to retain though unused in MPlus run
        included = [cluster, stratification, weight, id_variable] + list(f_ser.explode())
        other_vars = wh([c in included for c in data_column_names], 0, list(data_column_names))  # data variables to add
        ve = ve + [add_to_string_wrapped(other_vars, string='AUXILIARY =', end=';')]  # add auxiliary to extras statement
    if categorical is not None:
        v_stmnt = v_stmnt + '\n\tCATEGORICAL = %s;' % categ_vars  # add categorical statement (if need)
    if len(ve) > 0:
        v_stmnt = v_stmnt + '\n\t' + functools.reduce(lambda x, y: x + '\n\t' + y, ve)  # add any extras
    if na_flag is not None:
        v_stmnt = v_stmnt + '\n\tMISSING = %s;' % na_flag  # missing flag option
    if variable_statement_extras is not None:
        v_stmnt = v_stmnt + '\n\t' + variable_statement_extras
    if ESTIMATOR.lower() == 'bayes':
        if 'subpopulation' in v_stmnt.lower():
            warnings.warn('Subpopulation cannot be used with BAYES estimator. Switching to USEOBSERVATIONS.')
            v_stmnt = re.sub('subpopulation', 'USEOBSERVATIONS', v_stmnt, flags=re.IGNORECASE)

    # Analysis Statements
    est_stmnt = '\n\tESTIMATOR = %s;' % ESTIMATOR if ESTIMATOR is not None else ''  # estimator
    type_stmnt = '\n\tTYPE = %s;' % TYPE if TYPE is not None else ''  # type
    integ_stmnt = '\n\tINTEGRATION = MONTECARLO;' if ESTIMATOR.lower() == 'mlr' else ''  # integration
    if montecarlo is not None:
        integ_stmnt = re.sub('MONTECARLO', 'MONTECARLO(%d)' % montecarlo, integ_stmnt)  # MC points
    core_thread = '%d %d' % (cores, threads) if threads is not None else str(cores)
    proc_stmnt = '\n\tPROCESS = %s;' % core_thread if cores is not None else ''  # cores
    starts_stmnt = '\n\tSTARTS = %d;' % starts if starts is not None else ''  # random starts
    stit_stmnt = ' \n\tSTIT = %d;' % stit if stit is not None else ''  # starting iterations
    a_stmnt = '\nANALYSIS: %s%s%s%s%s%s' % (est_stmnt, type_stmnt, integ_stmnt, proc_stmnt, starts_stmnt, stit_stmnt)
    if interactive is not None:
        interact_file = interactive if type(interactive) is str else str(out_file) + '.txt'  # interactive file
        a_stmnt = a_stmnt + '\n\tINTERACTIVE = %s;' % interact_file  # interactive mode statement
    if ESTIMATOR.lower() == 'bayes':
        a_stmnt = a_stmnt + '\n\tBITERATIONS=(%s);' % bayes_biterations  # Bayes iterations
    if analysis_statement_extras is not None:
        a_stmnt = a_stmnt + '\n\t' + analysis_statement_extras

    # Model Statements
    space_i, space_fac = [' ', '@1; '] if free_load else [' ', '*; ']  # free or fix loading or variance
    s, fd = ['', '*'][free_load], dict(f_ser)  # * after first loading if free_load == True; dictionary version of f_ser
    by_str = '\n\t%s BY %s%s %s;'
    print(f_ser)
    by = [by_str % (f, f_ser[f][0], s if len(f_ser[f]) > 1 else '@1',  # '@1' if single indicator factor else '*' or ''
                    functools.reduce(lambda x, y: '%s %s' % (x, y), f_ser[f][1:]) if len(f_ser[f]) > 1
                    else f'; {f_ser[f][0]}@0') for f in fd]  # in case single-indicator factor
    by_stmnt = re.sub(' ;', ';', functools.reduce(lambda x, y: x + y, by))  # combine factor "by"s
    fac_var_stmnt = '\n\t' + functools.reduce(lambda x, y: x + y, [str(f + space_fac) for f in factors])  # variances
    if analysis.lower() == 'bifactor':  # if bifactor...
        by_bi_stmnt = add_to_string_wrapped(f_ser.explode(), string='\n\tGF BY ', end=';')  # paste & wrap bifactor BY
        fac_var_stmnt = fac_var_stmnt + 'GF@1; '  # add general factor variance to statement
        fac_cov_stmnt = '\n\tGF WITH ' + functools.reduce(lambda x, y: x + '@0 ' + y, factors) + '@0; '
        for f in range(len(factors) - 1):  # iterate over factors (except last, b/c all combinations will have been covered)
            others = functools.reduce(lambda x, y: x + '@0 ' + y, factors[f + 1:]) + '@0; '  # factors other than f @0
            fac_cov_stmnt = fac_cov_stmnt + '\n\t%s WITH ' % factors[f] + others  # factor f uncorrelated with others
        by_stmnt = by_bi_stmnt + by_stmnt
        fac_var_stmnt = fac_var_stmnt + fac_cov_stmnt
    m_stmnt = '\nMODEL:' + by_stmnt + fac_var_stmnt  # model statement
    if model_statement_extras is not None:
        m_stmnt = m_stmnt + '\n\t' + model_statement_extras

    # Results/Output Statements
    tech = 'TECH1 TECH3 TECH4'
    if ('mlr' in a_stmnt) or ('INTEGRATION' in a_stmnt):
        tech = tech + ' TECH8 TECH10'
    r_stnd = '\nOUTPUT: \n\t%s' % standardization + ' residual cinterval patterns ' + tech + ';'  # output
    if output_statement_extras is not None:
        r_stnd = r_stnd + '\n\t' + output_statement_extras
    r_plot = '\n\nPLOT: \n\tTYPE = plot1; \n\tTYPE = plot2; \n\tTYPE = plot3; \n\tMONITOR = off;'  # plot
    if plot_statement_extras is not None:
        r_plot = r_plot + '\n\t' + plot_statement_extras
    r_save = '\n\nSAVEDATA: \n\tFILE = \'%s.csv\'; \n\tSAVE = fscores; \n\tFORMAT = free;' % out_file  # save data
    if ESTIMATOR.lower() == 'bayes':  # BAYES estimator savedata options
        r_save = re.sub('SAVE = fscores', 'SAVE = fscores (%s %s)' % (tuple(bayes_fscores_imputations)), r_save)
        r_save = r_save + '\n\tFACTORS = %s;' % functools.reduce(lambda q, r: '%s %s' % (q, r), factors)
        if out_file != 0:
            r_save = r_save + '\n\tBPARAMETERS = %s_BPARAMETERS.dat;' % out_file
    if (ESTIMATOR.lower() in ['wlsmv', 'mlmv']) and (out_file != 0):  # file for later fit comparison (certain estimators)
        r_save = r_save + '\n\tDIFFTEST is %s_DIFFTEST.dat;' % out_file
    if save_statement_extras is not None:
        r_save = r_save + '\n\t' + save_statement_extras
    o_stmnt = r_stnd + r_plot + r_save if plots else r_stnd + r_save  # full output statement

    # Full Script & Write (if desired)
    script = functools.reduce(lambda x, y: x + '\n' + y, [d_stmnt, v_stmnt, a_stmnt, m_stmnt, o_stmnt])
    if write:
        script_file = open(str(out_file + '.inp'), 'w')
        script_file.writelines(script)
        script_file.close()

    return(script)  # return script & Mplus-ready data file


def read_mplus_lines(file, headers=['MODEL FIT INFORMATION', 'STANDARDIZED MODEL RESULTS'],
                     stop_header=None, read_next=False):
    """Read MPlus Text."""
    mplus_file = open('%s.out' % file, 'r')  # access MPlus output
    mplus_0, mplus_1 = [], []  # start empty lists
    start_0, start_1 = False, False  # don't start storing read output yet for sections
    for line in mplus_file:  # iterate through lines in text output
        if headers[0] in line:  # if first header specfied...
            start_0 = True  # signal to start reading chunk for 1st section
        elif headers[1] in line:
            start_0, start_1 = False, True  # turn off fit reading
        elif (start_1) and ('WITH' in line):
            break  # stop reading once done with both sections
        if start_0:
            mplus_0 = mplus_0 + [line]  # read output text for next section (if desired)
        if (start_1) and (read_next):
            mplus_1 = mplus_1 + [line]  # read in MPlus output text
    mplus_file.close()  # close file connection
    return mplus_0, mplus_1


def read_savedata(data, factors, ix, mplus_out_file, standardize=True, na_flag=None, join_data=False):
    """Read savedata from Mplus."""
    pandas2ri.deactivate()
    if ix in data.columns:
        data = data.set_index(ix)
    mplus_out_file = re.sub('[.]out$', '', mplus_out_file)  # ensure doesn't already have file extension
    mplus_output = mplus.readModels('%s.out' % mplus_out_file)
    ESTIMATOR = str(mplus_output.rx('input')[0].rx('analysis')[0].rx('estimator')[0][0])  # estimator
    na_flag = str(mplus_output.rx('input')[0].rx('variable')[0].rx('missing')[0][0])  # missingness symbol
    df = data.set_index(ix).rename_axis(ix) if ix in data.columns else data  # ensure data index set
    df = df.replace(na_flag, np.nan)
    cols = list(df.columns) + list(factors)  # desired column names (not truncated at 8 character & forced to uppercase by MPlus)
    m_cols = [c.upper()[:8] for c in cols]  # expected MPlus column names
    ef = pd.concat([df, pd.DataFrame(columns=[c.upper()[:8] for c in factors], index=df.index).rename_axis(ix)],
                   axis=1)  # data + NaN dataframe for factors
    sv = mplus_output.rx('savedata')[0]  # savedata & empty dataframe)
    sv_names = [str(i) for i in sv.names]
    if ESTIMATOR.lower() == 'bayes':  # parse variable names for BAYES estimator savedata
        # sv_names = wh([type(i) != rpy2.rinterface.NACharacterType for i in sv.names], 1, list(sv.names))
        bayes_i = mplus_output.rx('input')[0].rx('savedata')[0]
        bayes_f = wh([('SAVE' in i.upper()) and ('FSCORES' in i.upper()) for i in bayes_i], 1, [i for i in bayes_i])
        bit = int(re.sub('.*fscores.*[(]([0-9]*) .*', '\\1', bayes_f))  # factor score distribution # draws
        svnm = pd.Series([str(i) for i in sv.names]).apply(lambda c: [c] * bit if '+' in c else [c])
        svnm = functools.reduce(lambda x, y: wh([(i != 'NA') for i in x + y], 1, x + y), list(svnm))
        sv_names = [re.sub('_Mean', '', re.sub(' ', '_', s)) for s in svnm]
    print(sv_names)
    sv = pd.DataFrame(dict(zip(sv_names, [np.array(x) for x in sv]))).set_index(ix) if 'names' in dir(sv) else ef
    sv = sv[wh(['+' in c for c in sv.columns], 0, list(sv.columns))]
    for i, f in enumerate([x.upper()[:8] for x in factors]):
        sv.columns = [factors[i] if re.match('^%s$' % f, c) else c for c in sv.columns]
    try:
        saved = sv.rename(dict(zip(m_cols, cols)), axis=1)  # try to rename columns
    except Exception as err:
        saved = sv
        print(err, '\n\n%s\nCould not rename columns in savedata!\n%s' % ('=' * 80, '=' * 80))
    saved = saved.replace(na_flag, np.nan)  # replace NaN flags
    if (standardize) and (saved[factors].dropna().empty is False):  # if successfully extracted & want to standardize...
        saved.loc[:, factors] = sp.stats.zscore(saved[factors], nan_policy='omit')  # standardize (if wanted)
    if saved[factors].dropna().empty:
        warnings.warn('\n\n%s\n\nUnable to extract factor scores for %s\n\n%s' % ('*' * 80, mplus_out_file, '*' * 80))
    if join_data:
        print('\n\n%s\nJoining original and MPlus savedata...\n%s' % ('=' * 80, '=' * 80))
        saved = saved.join(df[wh([c in saved.columns for c in df.columns], 0, list(df.columns))])
        for c in saved.columns:
            if saved[[c]].shape[1] > 1:  # if duplicate columns, retain only one
                warnings.warn('\n\n%s\nJDuplicate column (%s). Using first.\n%s' % ('=' * 80, c, '=' * 80))
                yy = saved[c].iloc[:, 0]
                saved = saved.drop(c, axis=1)
                saved.loc[:, c] = yy
    return saved, mplus_output  # return savedata, output, & success indicator


def read_fit(model_names, mplus_out_files=None, mplus_output=None, digits=2):
    """Read Mplus fit statistics & loadings for MLR-Estimated factor analysis."""
    name_drop = ['Mplus.version', 'Title', 'AnalysisType', 'DataType', 'Estimator', 'NGroups', 'NDependentVars',
                 'NIndependentVars', 'NContinuousLatentVars', 'AICC', 'aBIC', 'LLCorrectionFactor', 'Filename']
    head = ('\n\n%s\n\n' % ('=' * 80), '\n\n%s\n' % ('=' * 80))
    fit = {}
    try:
        out = [mplus.readModels('%s.out' % i) for i in mplus_out_files] if mplus_output is None else mplus_output
        ESTIMATOR = str(mplus_output[0].rx('input')[0].rx('analysis')[0].rx('estimator')[0][0])  # estimator
        ft = [dict(zip(x.rx('summaries')[0].names, [i[0] for i in x.rx('summaries')[0]])) for x in out]
        ft = pd.DataFrame(ft).assign(Model=model_names).set_index('Model').rename({'Observations': 'N'}, axis=1)
        print('%sCheck fit index model names & file name alignment:%s' % head)
        [print([i[0], i[1]]) for i in zip(ft.index.values, ft.Filename)]  # so can see alignment b/t files & model names
        name_drop = wh([n in ft.columns for n in name_drop], 1, name_drop)  # only drop present undesired indices
        ft = ft.drop(name_drop, axis=1)  # drop irrelevant columns
        if ESTIMATOR.lower() == 'bayes':
            fit = ft.rename({'PostPred_PValue': 'Posterior-Predictive P-Value',
                             **dict(zip(['ObsRepChiSqDiff_95CI_%sB' % i for i in ['L', 'U']],
                                        ['Observed-Replicated Chi-Squared %s' % i for i in ['LB', 'UB']]))}, axis=1)
            fit = dict(fit)
        elif ('CFI' in ft.columns) or ('BIC' in ft.columns):  # if success in reading fit indices
            fit = dict(ft)
            if 'CFI' in ft.columns:
                rmsea_ci = (str('{:.3f}'.format(fit['RMSEA_90CI_LB'])),  # upper bound RMSEA CI
                            str('{:.3f}'.format(fit['RMSEA_90CI_UB'])))  # lower bound RMSEA CI
                rmsea = '{:.3f}'.format(fit['RMSEA_Estimate']) + ' [%s-%s]' % rmsea_ci  # RMSEA & CI
                chisq = (fit['ChiSqM_Value'], int(fit['ChiSqM_DF']))  # Chi-Squared & degrees of freedom
                fit.update({'Chi-Squared (df)': '%d (%d)' % chisq, 'RMSEA [90% CI]': rmsea})
        if 'Parameters' in ft.columns:
            fit.update({'# of Parameters': ['' if pd.isnull(i) else int(i) for i in fit['Parameters']]})
    except Exception as e:
        print(e)
        warnings.warn('\nWARNING: Read fit failed. Returning NaN.')
        fit = {'Index': np.nan}
    return fit  # fit statistics dictionary


def read_loadings(fac_series, model_names=None, mplus_output=None, mplus_files=None, all_indicators=None,
                  analysis='CFA', general_factor_name='GF', loading_type='stdyx.standardized',
                  headers=['MODEL FIT INFORMATION', 'STANDARDIZED MODEL RESULTS'], digits=3):
    """Read factor loadings from Mplus."""
    if model_names is None:
        model_names = list(fac_series.index)
    if all_indicators is None:
        fs = fac_series.apply(lambda x: [x[k] for k in x]) if fac_series[model_names[0]] is dict else fac_series
        all_indicators = list(pd.unique(fs.explode().explode()))
    factor_loadings = pd.DataFrame(index=all_indicators, columns=model_names)  # to hold loadings
    factor_loadings_GF = pd.DataFrame(index=all_indicators, columns=model_names)  # for GF loadings
    mplus_output = [mplus.readModels('%s.out' % m) for m in mplus_files] if mplus_output is None else mplus_output
    parameters = [wh(m.names, 'parameters', m) for m in mplus_output]
    for m in range(len(model_names)):
        try:
            factors = [general_factor_name] + fac_series[m] if analysis.lower() == 'bifactor' else fac_series[m]
            factors_mplus = [f.upper()[:8] for f in factors]  # MPlus capitalized & truncated factor names
            params = wh(parameters[m].names, loading_type, parameters[m])  # standardized fit statistics
            loads = pd.DataFrame([np.array(p) for p in params]).T.iloc[:, 0:3].replace(999, np.nan)  # subset columns
            loads = loads.rename({0: 'Factor', 1: 'Indicator', 2: 'Loading'}, axis=1)  # rename columns
            loads.loc[:, 'Factor'] = [re.sub('[.]BY$', '', f) for f in loads.Factor]  # erase .BY appended to factors
            loads = loads[loads.Factor.isin(factors_mplus)]
            fs = loads.Factor  # store factor column to iterate & replace capitalized factor names with originals (below)
            facs_o = [wh([i.upper() for i in factors], f, factors) if f not in factors else f for f in fs]
            loads.loc[:, 'Factor'] = facs_o  # replace with original factor names (see above)
            loads = loads.set_index(['Factor', 'Indicator'])
            if analysis.lower() == 'bifactor':
                gf_loads = loads.loc['GF']  # general factor loadings
            loads = loads.loc[fac_series[m]].reset_index('Factor', drop=True)  # specific factor loadings
            if analysis.lower() == 'bifactor':
                loads = loads.join(gf_loads, on='Indicator', rsuffix='_G')  # GF loads -> column
                loads.columns = ['Specific Factors', 'General Factors']  # rename columns to indicate general or specific
            factor_loadings.iloc[:, m] = loads.sort_index(axis=0, level='Indicator').iloc[:, 0]  # sort
            if analysis.lower() == 'bifactor':
                factor_loadings_GF.iloc[:, m] = loads.sort_index(axis=0, level='Indicator').iloc[:, 1]  # sort
        except Exception as e:
            print(e, '\n\n%s\n\nFailed to extract %s model loadings.\n\n%s' % ('*' * 80, model_names[m], '*' * 80))
    if analysis.lower() == 'bifactor':
        loads_gf, loads = factor_loadings, factor_loadings_GF
        loads = pd.concat([loads_gf.assign(Factor='General'),
                           loads.assign(Factor='Specific')], axis=0).reset_index()
        factor_loadings = loads.set_index(['Factor', 'index']).rename_axis(['Factor', 'Indicator'])
    return factor_loadings  # factor loadings dataframe
