import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import linregress, pearsonr
from pathlib import Path
from tqdm import tqdm
import joblib
import numpy as np
import pandas as pd

from specs import *

def gather_downstream(result_path):
    perf = {}
    for task_path in Path(result_path).glob('*'):
        res_path = Path(task_path, 'results.pkl')
        task = task_path.name
        if res_path.exists():
            metrics = joblib.load(res_path)
            m_key = downstream_scores[task]
            if len(metrics) == 2:
                m_val = metrics[-1][m_key]
            else:
                m_val = np.mean([m[m_key] for m in metrics[1:]])
        else:
            m_val = None
        perf[f'DOWNSTREAM_{task}'] = m_val
    return perf

def get_rsa(model_path, dataset='NH2015'):
    rsa_path = Path(model_path, f'RSA_{dataset}.pkl')
    if rsa_path.exists():
        rsa_data = joblib.load(rsa_path)
        rsa_mean = np.nanmean(rsa_data['subjects_r'])
        n_subjects = (~np.isnan(rsa_data['subjects_r'])).sum()
        rsa_se = np.nanstd(rsa_data['subjects_r'])/(np.sqrt(n_subjects-1))
        return {f'rsa_{dataset}_mean': rsa_mean, f'rsa_{dataset}_se': rsa_se}
    else:
        return None

def get_regression(model_path, dataset='NH2015'):
    reg_path = Path(model_path, f'REG_{dataset}.pkl')
    out = {}
    if reg_path.exists():
        reg_data = joblib.load(reg_path)
        for row in reg_data:
            out['REG_{}_{}'.format(dataset, row['voxel_id'])] = row['metrics'][0]
        return out
    else:
        return None

def load_results(results_path):
    all_data = []
    for model_path in tqdm(Path(results_path).glob('*')):
        model = model_path.stem
        nh2015_rsa = get_rsa(model_path, 'NH2015')
        b2021_rsa = get_rsa(model_path, 'B2021')
        nh2015_reg = get_regression(model_path, 'NH2015')
        nh2015c_reg = get_regression(model_path, 'NH2015comp')
        b2021_reg = get_regression(model_path, 'B2021')
        downstream_perf = gather_downstream(Path(model_path, 'downstream'))
        model_data = {'model': model}
        if nh2015_rsa is not None:
            model_data.update(nh2015_rsa)
        if nh2015_reg is not None:
            model_data.update(nh2015_reg)
        if nh2015c_reg is not None:
            model_data.update(nh2015c_reg)
        if b2021_rsa is not None:
            model_data.update(b2021_rsa)
        if b2021_reg is not None:
            model_data.update(b2021_reg)
        if downstream_perf is not None:
            model_data.update(downstream_perf)
        all_data.append(model_data)
        
    results_df = pd.DataFrame(all_data)
    
    def get_se(row):
        n_subjects = (~row[subj_cols].isna()).sum()
        if n_subjects == 0:
            return 0
        else:
            return np.nanstd(row[subj_cols]) / (np.sqrt(n_subjects-1))
    
    for d in ['NH2015', 'B2021']:
        voxel_data = pd.read_pickle(f'../data/neural/{d}/df_roi_meta.pkl')
        reg_cols = [c for c in results_df.columns if c.startswith(f'REG_{d}')]
        per_subj_cols = {}
        for s in voxel_data['subj_idx'].unique():
            s_df = voxel_data.loc[voxel_data['subj_idx']==s]
            s_cols = np.array(reg_cols)[s_df.index]
            results_df[f'REG_{d}_SUBJ_{s}'] = results_df.apply(lambda row: np.median(row[s_cols]), axis=1)
        subj_cols = [c for c in results_df.columns if c.startswith(f'REG_{d}_SUBJ')]
        results_df[f'REG_{d}_mean'] = results_df.apply(lambda row: np.nanmean(row[subj_cols]), axis=1)
        results_df[f'REG_{d}_se'] = results_df.apply(get_se, axis=1)

    downstream_cols = [c for c in results_df.columns if c.startswith('DOWNSTREAM')]
    for c in downstream_cols:
        results_df[c+'_zscore'] = (results_df[c] - results_df[c].mean()) / results_df[c].std()
    zscore_cols = [c+'_zscore' for c in downstream_cols]
    results_df['DOWNSTREAM_global'] = results_df.apply(lambda row: np.mean(row[zscore_cols]), axis=1)
    
    return results_df

def too_close(history, x, y, tolerance=[0.5, 0.02]):
    for h in history:
        if (abs(h[0] - x) < tolerance[0]) and (abs(h[1] - y) < tolerance[1]):
            offset_x = 0
            offset_y = -0.01
            return x + offset_x, y + offset_y
        
    return x, y
    
def make_correlation_plot(df, xcol, ycol, 
                          x_offset_pixels=10,
                          y_offset_pixels=8,
                          xlabel=None,
                          ylabel=None,
                          color_by='model',
                          annotate_uniques=True):
    df = df.loc[df['model'] != 'topline']
    x = df[xcol].values
    y = df[ycol].values
    if color_by=='model':
        colors = df['model'].apply(lambda x: model_to_color[x])
    elif color_by=='domain':
        colors = df['model'].apply(lambda x: domain_to_color[x])
    elif color_by is None:
        colors = df['model'].apply(lambda x: 'b')
    models = df['model'].values
    
    mask = ~np.logical_or(np.isnan(x), np.isnan(y))
    x = x[mask]
    y = y[mask]
    colors = colors[mask]
    models = models[mask]
    
    if xlabel is None:
        xlabel = xcol
    if ylabel is None:
        ylabel = ycol

    # regression + stats
    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    # prepare fit line
    best_fit_x = np.linspace(x.min(), x.max(), 200)
    best_fit_y = best_fit_x * slope + intercept

    ax = plt.gca()
    ax.plot(best_fit_x, best_fit_y, 'k--', label='Linear fit')
    ax.scatter(x, y, alpha=0.7, c=colors)
    h = []
    if annotate_uniques:
        for mi, xi, yi in zip(models, x, y):
            if mi in unique_labels:
                xi, yi = too_close(h, xi+0.03, yi)
                ax.text(xi, yi, unique_labels[mi])
                h.append((xi, yi))
        
    # force a draw so transforms are up-to-date (important if calling before show)
    fig = ax.get_figure()
    fig.canvas.draw()  

    # choose a small segment around the middle of the fit line to compute the display slope
    mid = len(best_fit_x)
    i0, i1 = max(0, mid - 6), min(len(best_fit_x) - 1, mid + 6)
    p0 = np.array([best_fit_x[i0], best_fit_y[i0]])
    p1 = np.array([best_fit_x[i1], best_fit_y[i1]])

    # transform those two data points to display (pixel) coordinates
    disp0 = ax.transData.transform(p0)
    disp1 = ax.transData.transform(p1)
    dx, dy = disp1 - disp0

    # compute angle in display coords (degrees)
    angle = np.degrees(np.arctan2(dy, dx))

    # midpoint in display coords
    disp_mid = (disp0 + disp1) / 2.0

    # make a small perpendicular offset (so text sits *above* the line)
    # perpendicular vector to (dx,dy) is (-dy, dx)
    norm = np.hypot(dx, dy)
    if norm < 1e-8:
        # fallback: vertical-ish line; no pixel offset
        disp_text = disp_mid
    else:
        perp = np.array([-dy, dx]) / norm
        parallel = np.array([dx, dy]) / norm
        disp_text = disp_mid + perp * y_offset_pixels + parallel * x_offset_pixels

    # convert back to data coords for placement
    x_text, y_text = ax.transData.inverted().transform(disp_text)

    txt = f'$r={r_value:.2f},\\ p={p_value:.2g}$'
    ax.text(
        x_text, y_text, txt,
        rotation=angle,
        rotation_mode='anchor',
        ha='right', va='center',
        fontsize=10,
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.2)
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=color, label=label)
    for label, color in legend_names.items()
]
    plt.legend(handles=handles, ncol=2, loc='lower right')
    plt.tight_layout()
    return ax

def make_barplot(df, ycol='REG_NH2015_mean', err=True, spectemp=True, topline=True, xlabel='$R^2$', xlim=0.55):
    fig, ax = plt.subplots(figsize=(10, 6), nrows=1, ncols=2)
    df = df.sort_values(by=ycol)
    if err:
        err_col = '_'.join(ycol.split('_')[:-1]) + '_se'
        err = df[err_col]
    else:
        err = None
    if spectemp:
        spectemp_row = df.loc[df['model'] == 'braindnn_spectemp_filters']
        df = df.loc[df['model'] != 'braindnn_spectemp_filters']
        ax[0].axvline(spectemp_row[ycol].values[0], c='gray')
        if err is not None:
            err = df[err_col]
            mean = spectemp_row[ycol].values[0]
            se = spectemp_row[err_col].values[0]
            ax[0].axvspan(mean-se, mean+se, color='gray', alpha=0.3)
    if topline:
        topline_row = df.loc[df['model'] == 'topline']
        df = df.loc[df['model'] != 'topline']
        if 'max' in ycol:
            dataset = ycol.split('_')[1].upper()
            ycol_top = f'rsa_{dataset}_mean'
            ycol_se = f'rsa_{dataset}_se'
            print(ycol_top)
            print(ycol_se)
        else:
            ycol_top = ycol
            ycol_se = err_col
        ax[0].axvline(topline_row[ycol_top].values[0], c='gray')
        if err is not None:
            err = df[ycol_se]
            mean = topline_row[ycol_top].values[0]
            se = topline_row[ycol_se].values[0]
            ax[0].axvspan(mean-se, mean+se, color='gray', alpha=0.3)
    else:
        df = df.loc[df['model'] != 'topline']
        if err is not None:
            err = df[err_col]
        
    ax[0].barh(y = df['model'].apply(lambda x: m_to_label[x]), 
               width = df[ycol],
               xerr = err,
               color = df['model'].apply(lambda x: assign_color(x, segment_cochdnn=False)))

    model_domains = ['mel256-ec-base', 'mel256-ec-base-as', 'mel256-ec-base-fma', 'mel256-ec-base-ll']
    df_domains = df.set_index('model').loc[model_domains]
    
    labels = ['Mixture', 'Audioset', 'FMA', 'LL']
    ax[1].barh(y=[19,18,17,16], 
               width = df_domains[ycol], 
               color=['#5e02c7']*4,
               xerr=df_domains[err_col])
    ax[1].set_xlim([0,xlim])
    ax[1].set_ylim([-3,21])
    ax[1].set_yticks(ticks=[19,18,17,16],labels=labels, fontsize=10)
    ax[1].axhline(15)
    
    ax[1].text(0.25,20,'Pretraining dataset', fontsize=10, ha='center')
    ax[1].text(0.25,14,'Finetuning', fontsize=10, ha='center')
    ax[1].text(0.25,7,'Iterative refinement', fontsize=10, ha='center')
    model_ft = ['BEATs_iter3', 'BEATs_iter3_finetuned_on_AS2M_cpt1', 'dasheng_base', 'dasheng_base_ft-as']
    df_ft = df.set_index('model').loc[model_ft]
    
    labels = ['BEATs', 'BEATs + FT','Dasheng', 'Dasheng + FT']
    xpos = [13,12,10,9]
    ax[1].barh(y=xpos, 
               width = df_ft[ycol],
               xerr = df_ft[err_col],
               color=['#f5b042','#f5b042','#42f584','#42f584'],
               hatch=['','/','','/'])
    ax[1].set_yticks(ticks=[13.5,11.5],labels=['BEATs','Dasheng'], fontsize=10)
    ax[1].axhline(8)
    
    custom_legend = [
        Patch(facecolor='b', label='Base'),
        Patch(facecolor='r', label='Finetuned'),
    ]
    
    model_iter = ['BEATs_iter1','BEATs_iter2','BEATs_iter3','mel256-ec-base', 'mel256-ec-base_st-nopn', 'mel256-ec-large', 'mel256-ec-large_st-nopn']
    df_iter = df.set_index('model').loc[model_iter]
    
    labels = ['BEATs It1','BEATs It2','BEATs It3', 'EnCodecMAE Base It1', 'EnCodecMAE Base It2','EnCodecMAE Large It1', 'EnCodecMAE Large It2']
    xpos = [6,5,4,2,1,-1,-2]
    ax[1].barh(y=xpos, 
               width = df_iter[ycol],
               xerr = df_iter[err_col],
               color=['#f5b042','#f5b042','#f5b042','#5e02c7','#5e02c7','#5e02c7','#5e02c7'],
               hatch=['','|','||','','|','','|'])
    
    custom_legend = [
        Patch(facecolor='b', label='1'),
        Patch(facecolor='g', label='2'),
        Patch(facecolor='r', label='3')
    ]
    
    ax[1].set_yticks(ticks=[19,18,17,16,13,12,10,9,6,5,4,2,1,-1,-2],
                     labels=['Mixture',
                             'Audioset',
                             'Music (FMA)',
                             'Speech (LL)',
                             'BEATs',
                             'BEATs FT',
                             'Dasheng',
                             'Dasheng FT',
                             'BEATs (Iter 1)',
                             'BEATs (Iter 2)',
                             'BEATs (Iter 3)',
                             'Mel256→EC (Iter 1)',
                             'Mel256→EC (Iter 2)',
                             'Mel256→EC Large (Iter 1)',
                             'Mel256→EC Large (Iter 2)'])
    
    leg_row1_handles = [
        Patch(facecolor='#f5b042', label='BEATs'),
        Patch(facecolor='#5e02c7', label='EnCodecMAE'),
        Patch(facecolor='#42f584', label='Dasheng'),
        Patch(facecolor='#e3a6ab', label='Others')
        ]
    leg_row2_handles = [
        Patch(facecolor='white', edgecolor='black', hatch='/', label='Finetuned'),
        Patch(facecolor='white', edgecolor='black', hatch='|', label='Iter 2'),
        Patch(facecolor='white', edgecolor='black', hatch='||', label='Iter 3'),
    ]
    leg1 = fig.legend(handles=leg_row1_handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.03),bbox_transform=fig.transFigure)
    leg2 = fig.legend(handles=leg_row2_handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.075),bbox_transform=fig.transFigure)

    ax[0].set_xlabel(xlabel)
    ax[1].set_xlabel(xlabel)
    plt.tight_layout()
    
def layerwise_rsa(folder, model):
    out = {}
    for d in ['NH2015', 'B2021']:
        p = Path(folder, model, f'RSA_{d}.pkl')
        if p.exists():
            per_layer_rsa = joblib.load(p)
            data = pd.DataFrame([x for x in per_layer_rsa['subjects_r']])
            df_layerwise = data.groupby('layer').mean()
            df_layerwise['r_std'] = data.groupby('layer')['subj_r'].std()
            lw = [df_layerwise.loc[l]['subj_r'] for l in layer_map[model]]
            lw_std = [df_layerwise.loc[l]['r_std'] for l in layer_map[model]]
            out[f'lw_{d}'] = lw
            out[f'lw_std_{d}'] = lw_std
        else:
            out[f'lw_{d}'] = None
            out[f'lw_std_{d}'] = None
    out['model'] = model
    return out

def plot_components(df):
    cols_comp = [f'REG_NH2015comp_{i}' for i in range(6)]
    comp_data = df.set_index('model')
    comp_data = comp_data[cols_comp]
    comp_data = comp_data.loc[comp_data.index.map(lambda x: x!='topline')]
    component_names = ['LF', 'HF', 'Broadband', 'Pitch', 'Speech', 'Music']
    fig, ax = plt.subplots(figsize=(13,13), nrows=2, ncols=3)
    for i,c in enumerate(cols_comp):
        comp_i = comp_data[c].sort_values(ascending=True)
        spec = comp_i.loc['braindnn_spectemp_filters']
        comp_i = comp_i.drop(index='braindnn_spectemp_filters')
        ax[i//3,i%3].barh(y=comp_i.index.map(lambda x: m_to_label[x]), 
                          width=comp_i.values,
                          color=comp_i.index.map(lambda x: assign_color(x, segment_cochdnn=False)))
        ax[i//3,i%3].axvline(spec, c='gray')
        ax[i//3,i%3].set_title(component_names[i])
        ax[i//3,i%3].set_xlabel(r'$R^2$')
    plt.tight_layout()