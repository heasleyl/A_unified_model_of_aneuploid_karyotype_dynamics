import numpy as np
import pandas as pd
from scipy import optimize, stats


## General stats

def plot_pval_symbol(p):
    if p<=0.001:
        pdisplay = '***'
    elif p<=0.01:
        pdisplay = '**'
    elif p<=0.05:
        pdisplay = '*'
    else:
        pdisplay = 'ns'
    return pdisplay

def plot_pval_text(p):
    if p<=0.001:
        pdisplay = f'{p:.1e}'.split('e')
        pdisplay = pdisplay[0]+r'$\times10^{'+f'{int(pdisplay[1])}'+r'}$'
    elif p<=0.01:
        pdisplay = f'{p:.4f}'
    elif p<=0.1:
        pdisplay = f'{p:.3f}'
    elif p<=1:
        pdisplay = f'{p:.2f}'
    elif p>1:
        pdisplay = f'{p:.1f}'
    else:
        pdisplay = None
    return pdisplay


## Growth curves


def growth_to_inhibition(growth):
	## growth must be a Series with concentration as index
	uninhibited = growth.loc[0]
	if type(uninhibited) == pd.Series:
		uninhibited = uninhibited.mean()
	return 1-(growth/uninhibited)


def hill_equation(c, e, h):
    # c: compound/ligand concentration (x-axis)
    # e: EC50, IC50 or KA, depending on preference
    # h: hill coefficient
    return 1/(1+(e/c)**h)


def logistic(t, A, mu, l):
    return A/(1+np.exp((4*mu/A)*(l-t) + 2))


def linear_interpolation(x):
    return np.linspace(x.iloc[0], x.iloc[-1], x.shape[0])


def parse_gc(dat_path, samples_path, dataset_tag, Tend=24, skiprows=30, blank_correct=True, add_tags=None):

    samples = pd.read_csv(samples_path)
  
    gc_dat = pd.read_csv(dat_path, sep='\t', skiprows=skiprows)
    gc_dat.columns = ['time', 'temp'] + list(gc_dat.columns[2:])
    gc_dat['time'] = pd.to_timedelta(gc_dat['time'])/pd.Timedelta('1h')
    gc_dat = pd.melt(gc_dat, id_vars=['time', 'temp'], var_name='well', value_name='OD600')
    gc_dat = gc_dat.merge(samples, on='well')
    if blank_correct:
        blank = np.median(gc_dat.loc[(gc_dat['strain']=='blank') & (gc_dat['time']<3), 'OD600'])
        gc_dat['OD600_blank'] = gc_dat['OD600'] - blank
        
    gc_dat = gc_dat.loc[gc_dat['time']<=Tend]
    gc_dat['dataset'] = dataset_tag

    if type(add_tags) == dict:
        for k,v in add_tags.items():
            gc_dat[k] = v

    return samples, gc_dat


def compute_gc_auc(gc_dat, samples, dataset, od='OD600', average_reps=None):

    gc_auc = gc_dat.sort_values(by='time').groupby('well')\
    .apply(lambda x: (x[od]-x.iloc[0][od]).sum(), include_groups=False)\
    .rename('auc').reset_index()
    gc_auc = gc_auc.merge(samples, on='well')
    gc_auc['dataset'] = dataset
    if type(average_reps) == list:
        gc_auc_mean = gc_auc.groupby(average_reps+['dataset'])['auc'].mean().rename('mean_auc').reset_index()
        return gc_auc, gc_auc_mean
    else:
        return gc_auc


def compute_gc_logistic(gc_dat, samples, dataset, od='OD600_blank', average_reps=None):
    gc_logistic = []
    for w, df in gc_dat.sort_values(by='time').groupby('well'):
        dat = df.set_index('time')[od]
        try:
            A, mu, l = optimize.curve_fit(logistic, dat.index, dat)[0]
            fit = True
        except RuntimeError:
            A, mu, l = (0, 0, np.inf)
            fit = False
        gc_logistic.append([w, A, mu, l, fit])
    
    gc_logistic = pd.DataFrame(gc_logistic, columns=['well', 'A', 'mu', 'lag', 'fit'])
    gc_logistic['dataset'] = dataset
    gc_logistic = gc_logistic.merge(samples, on='well')
    
    if type(average_reps) == list:
        gc_logistic_mean = gc_logistic.loc[gc_logistic['fit']==True].groupby(average_reps+['dataset'])\
        [['A', 'mu', 'lag']].mean()\
        .rename({'A':'mean_A', 'mu':'mean_mu', 'lag':'mean_lag'}, axis=1).reset_index()
        return gc_logistic, gc_logistic_mean
    
    else:
        return gc_logistic


def compute_gc_maxslope(gc_dat, samples, dataset, wdw=5, od='OD600', average_reps=None):

    gc_maxslope = []
    for w, df in gc_dat.sort_values(by='time').groupby('well'):
        dat = df.set_index('time')[od]

        time_values = dat.index.sort_values().values
        slope_matrix = []
        for i in np.arange(time_values.shape[0]-wdw+1):
            wd_time_values = time_values[i:i+wdw]
            wd_dat = dat.loc[wd_time_values]
            lr = stats.linregress(wd_time_values, wd_dat)
            slope_matrix.append([w, lr.slope, lr.intercept, lr.rvalue, np.median(wd_time_values)])
        
        slope_matrix = pd.DataFrame(slope_matrix, columns=['well', 'slope', 'intercept', 'rvalue', 'wdmid'])
        gc_maxslope.append(slope_matrix.sort_values(by='slope', ascending=False).iloc[0])
    
    gc_maxslope = pd.concat(gc_maxslope, axis=1).T
    gc_maxslope['dataset'] = dataset
    gc_maxslope = gc_maxslope.merge(samples, on='well')
        
    if type(average_reps) == list:
        gc_maxslope_mean = gc_maxslope.groupby(average_reps+['dataset'])\
        [['slope', 'intercept', 'rvalue', 'wdmid']].mean()\
        .rename({'slope':'mean_slope', 'intercept':'mean_intercept', 'rvalue':'mean_rvalue', 'wdmid':'mean_wdmid'}, axis=1).reset_index()
        return gc_maxslope, gc_maxslope_mean
    else:
        return gc_maxslope