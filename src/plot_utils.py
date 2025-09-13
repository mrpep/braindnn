import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress, pearsonr
from pathlib import Path
from tqdm import tqdm
import joblib
import numpy as np
import pandas as pd

from specs import downstream_scores

def gather_downstream(result_path):
    perf = {}
    for task_path in Path(result_path).glob('*'):
        res_path = Path(task_path, 'results.pkl')
        if res_path.exists():
            metrics = joblib.load(res_path)
            task = task_path.name
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
        rsa_se = np.nanstd(rsa_data['subjects_r'])/(np.sqrt(len(rsa_data['subjects_r'])-1))
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
        downstream_perf = gather_downstream(Path(model_path, 'downstream'))
        model_data = {'model': model}
        model_data.update(nh2015_rsa)
        model_data.update(nh2015_reg)
        model_data.update(nh2015c_reg)
        model_data.update(b2021_rsa)
        model_data.update(downstream_perf)
        all_data.append(model_data)
        
    results_df = pd.DataFrame(all_data)
    voxel_data = pd.read_pickle('../data/neural/NH2015/df_roi_meta.pkl')

    reg_cols = [c for c in results_df.columns if c.startswith('REG_NH2015')]
    per_subj_cols = {}
    for s in voxel_data['subj_idx'].unique():
        s_df = voxel_data.loc[voxel_data['subj_idx']==s]
        s_cols = np.array(reg_cols)[s_df.index]
        results_df[f'REG_NH2015_SUBJ_{s}'] = results_df.apply(lambda row: np.median(row[s_cols]), axis=1)
    subj_cols = [c for c in results_df.columns if c.startswith('REG_NH2015_SUBJ')]
    results_df['REG_NH2015_mean'] = results_df.apply(lambda row: np.mean(row[subj_cols]), axis=1)

    downstream_cols = [c for c in results_df.columns if c.startswith('DOWNSTREAM')]
    for c in downstream_cols:
        results_df[c+'_zscore'] = (results_df[c] - results_df[c].mean()) / results_df[c].std()
    zscore_cols = [c+'_zscore' for c in downstream_cols]
    results_df['DOWNSTREAM_global'] = results_df.apply(lambda row: np.mean(row[zscore_cols]), axis=1)
    
    return results_df

def make_correlation_plot(df, xcol, ycol, 
                          offset_pixels=8,
                          xlabel=None,
                          ylabel=None):
    x = df[xcol].values
    y = df[ycol].values
    
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
    ax.scatter(x, y, alpha=0.7)

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
        disp_text = disp_mid + perp * offset_pixels

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
    plt.tight_layout()
    return ax