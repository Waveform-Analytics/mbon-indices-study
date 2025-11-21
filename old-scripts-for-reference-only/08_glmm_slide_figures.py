#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from mbon_pipeline.core.paths import ProjectPaths


ACOUSTIC_INDICES = ['BI', 'EAS', 'EPS_KURT', 'EVNtMean', 'nROI']
ENV_COVARS = ['temp', 'depth']


def get_fig_dir(paths: ProjectPaths) -> Path:
    fig_dir = paths.get_figures_dir()
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def load_fixed_effects(paths: ProjectPaths) -> pd.DataFrame:
    glmm_dir = paths.processed_data / 'glmm_results'
    combined = glmm_dir / 'glmm_all_coefficients.csv'
    if combined.exists():
        df = pd.read_csv(combined)
    else:
        frames = []
        for f in glmm_dir.glob('glmm_coef_*.csv'):
            resp = f.stem.replace('glmm_coef_', '')
            d = pd.read_csv(f)
            d['response'] = resp
            frames.append(d)
        if frames:
            df = pd.concat(frames, ignore_index=True)
        else:
            raise FileNotFoundError('No fixed effects coefficient files found')
    df = df[df['effect'] == 'fixed'] if 'effect' in df.columns else df
    df = df[df['term'] != '(Intercept)']
    return df


def plot_forest_fixed_effects(paths: ProjectPaths) -> Path:
    df = load_fixed_effects(paths)
    df = df[df['term'].isin(ACOUSTIC_INDICES + ENV_COVARS)]
    responses = sorted(df['response'].unique())

    # Figure layout (16:9) with small multiples
    n = len(responses)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12.8, 7.2), sharex=False)
    axes = np.array(axes).reshape(-1)

    palette = sns.color_palette('RdBu_r', 11)

    for i, resp in enumerate(responses):
        ax = axes[i]
        sub = df[df['response'] == resp].copy()
        # Order by absolute estimate descending for readability
        sub['abs_est'] = sub['estimate'].abs()
        sub = sub.sort_values('abs_est', ascending=True)
        ax.hlines(y=range(len(sub)), xmin=sub['conf.low'], xmax=sub['conf.high'], color='#444', linewidth=1.5)
        sc = ax.scatter(sub['estimate'], range(len(sub)), c=sub['estimate'], cmap='RdBu_r', s=40, vmin=-sub['abs_est'].max(), vmax=sub['abs_est'].max())
        ax.axvline(0, color='#777', linewidth=1)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(sub['term'])
        ax.set_xlabel('Estimate (β)')
        ax.set_title(resp, pad=6, fontsize=12)
        ax.tick_params(axis='both', labelsize=10)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    out = get_fig_dir(paths) / '08_forest_fixed_effects.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return out


def plot_delta_aic(paths: ProjectPaths) -> Path:
    stats_path = paths.processed_data / 'glmm_results' / 'glmm_model_stats_both_approaches.csv'
    df = pd.read_csv(stats_path)
    df = df[['approach', 'response', 'AIC']].copy()
    # ΔAIC vs best per response
    df['deltaAIC'] = df.groupby('response')['AIC'].transform(lambda x: x - x.min())
    # Order responses by mean deltaAIC to pack bars nicely
    order = df.groupby('response')['deltaAIC'].mean().sort_values(ascending=False).index.tolist()
    df['response'] = pd.Categorical(df['response'], categories=order, ordered=True)

    # Plot
    approaches = sorted(df['approach'].unique())
    colors = {'universal': '#3498db'}
    for a in approaches:
        if a not in colors:
            colors[a] = '#e67e22'

    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    for a in approaches:
        sub = df[df['approach'] == a]
        ax.barh(sub['response'], sub['deltaAIC'], label=a, color=colors[a], alpha=0.85)
    ax.set_xlabel('ΔAIC (vs best per response)')
    ax.set_ylabel('Response')
    ax.legend(frameon=False, fontsize=11)
    ax.grid(axis='x', alpha=0.3)
    ax.tick_params(axis='both', labelsize=11)

    plt.tight_layout()
    out = get_fig_dir(paths) / '08_model_comparison_aic.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return out


def create_summary(fig_paths: list, paths: ProjectPaths) -> Path:
    lines = []
    lines.append('# Figure Summary\n')
    for p in fig_paths:
        fname = p.name
        if 'forest' in fname:
            title = 'Fixed Effects (Forest) by Response'
            caption = 'Estimated effects and 95% CI for acoustic indices and environment.'
            notes = (
                'Shows β estimates (points) with confidence intervals (lines) for selected predictors. '\
                'Red/blue indicate direction (increase/decrease) on the link scale. Patterns across responses '
                'demonstrate indices contribute predictive signal beyond temperature, depth, month, and hour. '
                'Use this to discuss which indices align with biological vs anthropogenic responses.'
            )
        elif 'model_comparison_aic' in fname:
            title = 'Model Comparison (ΔAIC)'
            caption = 'ΔAIC per response comparing universal vs taxa-specific index sets.'
            notes = (
                'Bars show AIC differences relative to the best approach for each response (lower is better). '
                'Interpret per response only; avoids cross-response AIC comparisons. Useful as a robustness check '
                'that core conclusions are stable across index selection strategies.'
            )
        else:
            title = 'Figure'
            caption = 'Slide-ready figure.'
            notes = 'Use axis labels for context; add your own slide titles.'

        lines.append(f'## {title}\n')
        lines.append(f'- Filename: `{fname}`\n')
        lines.append(f'- Caption: {caption}\n')
        lines.append(f'- Notes: {notes}\n')

    out_md = get_fig_dir(paths) / 'FIGURE_SUMMARY.md'
    out_md.write_text('\n'.join(lines))
    return out_md


def main():
    paths = ProjectPaths()
    fig1 = plot_forest_fixed_effects(paths)
    fig2 = plot_delta_aic(paths)
    summary = create_summary([fig1, fig2], paths)
    print('Saved:')
    print(fig1)
    print(fig2)
    print(summary)


if __name__ == '__main__':
    main()