#!/usr/bin/env python3
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from mbon_pipeline.core.paths import ProjectPaths


ACOUSTIC_INDICES = ['BI', 'EAS', 'EPS_KURT', 'EVNtMean', 'nROI']
ENV_COVARS = ['temp', 'depth']
PRESENCE_RESPONSES = ['fish_present', 'dolphin_present', 'vessel_present']
COUNT_RESPONSES = ['fish_richness', 'fish_activity', 'dolphin_activity', 'dolphin_whistles', 'dolphin_burst_pulses', 'dolphin_echolocation']


def load_universal_coefficients(paths):
    glmm_dir = paths.processed_data / 'glmm_results'
    coefs_path = glmm_dir / 'glmm_all_coefficients_both_approaches.csv'
    if not coefs_path.exists():
        # Fallback to individual files
        frames = []
        for resp in PRESENCE_RESPONSES + ['fish_richness', 'fish_activity', 'dolphin_activity', 'dolphin_whistles', 'dolphin_burst_pulses', 'dolphin_echolocation', 'vessel_count']:
            f = glmm_dir / f'glmm_coef_{resp}.csv'
            if f.exists():
                df = pd.read_csv(f)
                df['response'] = resp
                df['approach'] = 'universal'
                frames.append(df)
        if not frames:
            raise FileNotFoundError('GLMM coefficient files not found')
        coefs = pd.concat(frames, ignore_index=True)
    else:
        coefs = pd.read_csv(coefs_path)
    # Keep fixed effects
    if 'effect' in coefs.columns:
        coefs = coefs[coefs['effect'] == 'fixed']
    return coefs


def create_coefficient_heatmap(paths):
    coefs = load_universal_coefficients(paths)
    # Filter predictors of interest
    predictors = ACOUSTIC_INDICES + ENV_COVARS
    coefs = coefs[coefs['term'].isin(predictors)]
    if 'approach' in coefs.columns:
        coefs = coefs[coefs['approach'] == 'universal']

    # Compute significance by CI excluding 0 if available
    if {'conf.low', 'conf.high'}.issubset(set(coefs.columns)):
        coefs['significant'] = (coefs['conf.low'] > 0) | (coefs['conf.high'] < 0)
    else:
        coefs['significant'] = False

    pivot = coefs.pivot_table(index='term', columns='response', values='estimate', aggfunc='mean')

    # Order rows: indices first, then environmental
    order = [p for p in ACOUSTIC_INDICES if p in pivot.index] + [p for p in ENV_COVARS if p in pivot.index]
    pivot = pivot.loc[order]

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, cmap='RdBu_r', center=0, cbar_kws={'label': 'Coefficient'})
    plt.title('GLMM Coefficients: Indices vs Responses (Universal Approach)')
    plt.xlabel('Response Variable')
    plt.ylabel('Predictor')
    out_path = paths.get_figure_path('07_coefficients_heatmap.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    return out_path


def load_final_indices(paths):
    f = paths.processed_data / 'indices_final_vif_checked_v2.json'
    if f.exists():
        with open(f, 'r') as fh:
            data = json.load(fh)
        return data.get('final_indices', ACOUSTIC_INDICES)
    return ACOUSTIC_INDICES


def _family_for_response(resp):
    if resp.endswith('_present'):
        return 'binomial'
    return 'poisson'


def _link_inverse(eta, family):
    if family == 'binomial':
        return 1.0 / (1.0 + np.exp(-eta))
    return np.exp(eta)


def plot_marginal_effects(paths):
    coefs = load_universal_coefficients(paths)
    if 'approach' in coefs.columns:
        coefs = coefs[coefs['approach'] == 'universal']
    final_idxs = load_final_indices(paths)
    # Choose a top index present in coefficients
    chosen = None
    for idx in ['nROI', 'EVNtMean', 'BI', 'EAS', 'EPS_KURT']:
        if idx in final_idxs and (coefs['term'] == idx).any():
            chosen = idx
            break
    if chosen is None:
        idx_counts = coefs['term'].value_counts()
        chosen = idx_counts.index[0]

    x = np.linspace(-2.0, 2.0, 100)  # standardized index range
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

    for ax, resp in zip(axes, PRESENCE_RESPONSES):
        df = coefs[coefs['response'] == resp]
        # Intercept
        b0 = df[df['term'] == '(Intercept)']['estimate']
        b0 = float(b0.iloc[0]) if len(b0) else 0.0
        # Index coefficient
        bx = df[df['term'] == chosen]['estimate']
        bx = float(bx.iloc[0]) if len(bx) else 0.0
        eta = b0 + bx * x
        y = _link_inverse(eta, _family_for_response(resp))
        ax.plot(x, y, color='#2c3e50')
        ax.set_title(resp)
        ax.set_xlabel(chosen)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel('Predicted probability')
    fig.suptitle(f'Marginal Effects: {chosen} vs Presence Responses (Universal GLMM)')
    out_path = paths.get_figure_path('07_marginal_effects.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    return out_path


def create_magnitude_bars(paths):
    glmm_dir = paths.processed_data / 'glmm_results'
    frames = []
    # Use individual response coef files for robustness
    for resp in ['fish_present', 'fish_richness', 'fish_activity', 'dolphin_present', 'dolphin_activity', 'dolphin_whistles', 'dolphin_burst_pulses', 'dolphin_echolocation', 'vessel_present']:
        f = glmm_dir / f'glmm_coef_{resp}.csv'
        if not f.exists():
            continue
        df = pd.read_csv(f)
        df = df[df['effect'] == 'fixed']
        df['response'] = resp
        frames.append(df[['response', 'term', 'estimate']])
    if not frames:
        # Fallback to combined coefficients filtered to universal
        df = load_universal_coefficients(paths)
        if 'approach' in df.columns:
            df = df[df['approach'] == 'universal']
        df = df[df['effect'] == 'fixed'] if 'effect' in df.columns else df
        frames = [df[['response', 'term', 'estimate']]]
    coef_all = pd.concat(frames, ignore_index=True)

    results = []
    for resp in coef_all['response'].unique():
        sub = coef_all[coef_all['response'] == resp]
        idx_mean = sub[sub['term'].isin(ACOUSTIC_INDICES)]['estimate'].abs().mean()
        env_mean = sub[sub['term'].isin(ENV_COVARS)]['estimate'].abs().mean()
        results.append({'response': resp, 'indices': idx_mean, 'environmental': env_mean})
    bar_df = pd.DataFrame(results)

    x = np.arange(len(bar_df))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width/2, bar_df['indices'], width, label='Indices', color='#3498db')
    ax.bar(x + width/2, bar_df['environmental'], width, label='Environmental', color='#e67e22')
    ax.set_xticks(x)
    ax.set_xticklabels(bar_df['response'], rotation=30, ha='right')
    ax.set_ylabel('Mean |Coefficient|')
    ax.set_title('Effect Magnitudes: Acoustic Indices vs Environmental Covariates')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    out_path = paths.get_figure_path('07_effect_magnitude_bars.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    return out_path


def create_odds_ratio_bars(paths):
    coefs = load_universal_coefficients(paths)
    if 'approach' in coefs.columns:
        coefs = coefs[coefs['approach'] == 'universal']
    coefs = coefs[coefs['effect'] == 'fixed'] if 'effect' in coefs.columns else coefs
    coefs = coefs[coefs['response'].isin(PRESENCE_RESPONSES)]
    rows = []
    for resp in PRESENCE_RESPONSES:
        df = coefs[coefs['response'] == resp]
        for idx in ACOUSTIC_INDICES:
            est = df[df['term'] == idx]['estimate']
            if len(est):
                orv = float(np.exp(est.iloc[0]))
                rows.append({'response': resp, 'index': idx, 'odds_ratio': orv})
    if not rows:
        return paths.get_figure_path('07_odds_ratio_bars.png')
    plot_df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, resp in zip(axes, PRESENCE_RESPONSES):
        sub = plot_df[plot_df['response'] == resp]
        ax.bar(sub['index'], sub['odds_ratio'], color='#2ecc71')
        ax.axhline(1.0, color='black', linewidth=1)
        ax.set_title(resp)
        ax.set_xticklabels(sub['index'], rotation=30, ha='right')
    axes[0].set_ylabel('Odds Ratio (exp(beta))')
    fig.suptitle('Presence Models: Index Odds Ratios (Universal GLMM)')
    out_path = paths.get_figure_path('07_odds_ratio_bars.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    return out_path


def create_delta_probability_bars(paths):
    coefs = load_universal_coefficients(paths)
    if 'approach' in coefs.columns:
        coefs = coefs[coefs['approach'] == 'universal']
    coefs = coefs[coefs['effect'] == 'fixed'] if 'effect' in coefs.columns else coefs
    coefs = coefs[coefs['response'].isin(PRESENCE_RESPONSES)]
    rows = []
    for resp in PRESENCE_RESPONSES:
        df = coefs[coefs['response'] == resp]
        b0 = df[df['term'] == '(Intercept)']['estimate']
        b0 = float(b0.iloc[0]) if len(b0) else 0.0
        for idx in ACOUSTIC_INDICES:
            bx = df[df['term'] == idx]['estimate']
            if not len(bx):
                continue
            bx = float(bx.iloc[0])
            p_low = 1.0 / (1.0 + np.exp(-(b0 + bx * (-1.0))))
            p_high = 1.0 / (1.0 + np.exp(-(b0 + bx * (1.0))))
            rows.append({'response': resp, 'index': idx, 'delta_p': p_high - p_low})
    if not rows:
        return paths.get_figure_path('07_delta_probability_bars.png')
    plot_df = pd.DataFrame(rows)
    x = np.arange(len(ACOUSTIC_INDICES))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, resp in zip(axes, PRESENCE_RESPONSES):
        sub = plot_df[plot_df['response'] == resp]
        ax.bar(sub['index'], sub['delta_p'], color='#9b59b6')
        ax.set_title(resp)
        ax.set_xticklabels(sub['index'], rotation=30, ha='right')
    axes[0].set_ylabel('ΔP (index: -1 → +1)')
    fig.suptitle('Presence Models: Δ Predicted Probability by Index')
    out_path = paths.get_figure_path('07_delta_probability_bars.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    return out_path


def _predictor_sd_map(paths, approach):
    # Build std-dev map for predictors per response group
    sd = {}
    if approach == 'universal':
        df = pd.read_csv(paths.processed_data / 'model_data_universal.csv')
        for col in df.columns:
            if col in ACOUSTIC_INDICES + ENV_COVARS:
                sd[col] = float(pd.to_numeric(df[col], errors='coerce').std())
    else:
        # taxa-specific: compute per group
        for group, fname in {'fish': 'model_data_fish.csv', 'dolphin': 'model_data_dolphin.csv', 'vessel': 'model_data_vessel.csv'}.items():
            df = pd.read_csv(paths.processed_data / fname)
            gmap = {}
            for col in df.columns:
                if col in ACOUSTIC_INDICES + ENV_COVARS:
                    gmap[col] = float(pd.to_numeric(df[col], errors='coerce').std())
            sd[group] = gmap
    return sd


def create_normalized_effect_panels(paths, approach='universal'):
    coefs = load_universal_coefficients(paths)
    coefs = coefs[coefs['effect'] == 'fixed'] if 'effect' in coefs.columns else coefs
    if approach == 'universal' and 'approach' in coefs.columns:
        coefs = coefs[coefs['approach'] == 'universal']
    sdmap = _predictor_sd_map(paths, approach)

    def sd_for_term(term, response):
        if approach == 'universal':
            return sdmap.get(term, 1.0)
        # map response to group
        group = 'fish' if response.startswith('fish_') else ('dolphin' if response.startswith('dolphin_') else 'vessel')
        return sdmap.get(group, {}).get(term, 1.0)

    rows_presence = []
    for resp in [r for r in PRESENCE_RESPONSES if (coefs['response'] == r).any()]:
        sub = coefs[coefs['response'] == resp]
        idx_est = []
        env_est = []
        for _, row in sub.iterrows():
            term = row['term']
            beta = float(row['estimate'])
            sdx = sd_for_term(term, resp)
            if term in ACOUSTIC_INDICES:
                idx_est.append(np.exp(beta * sdx))
            elif term in ENV_COVARS:
                env_est.append(np.exp(beta * sdx))
        rows_presence.append({
            'response': resp,
            'indices_multiplier': float(np.mean(idx_est)) if idx_est else np.nan,
            'env_multiplier': float(np.mean(env_est)) if env_est else np.nan,
        })

    rows_counts = []
    for resp in [r for r in COUNT_RESPONSES if (coefs['response'] == r).any()]:
        sub = coefs[coefs['response'] == resp]
        idx_est = []
        env_est = []
        for _, row in sub.iterrows():
            term = row['term']
            beta = float(row['estimate'])
            sdx = sd_for_term(term, resp)
            if term in ACOUSTIC_INDICES:
                idx_est.append(np.exp(beta * sdx))
            elif term in ENV_COVARS:
                env_est.append(np.exp(beta * sdx))
        rows_counts.append({
            'response': resp,
            'indices_multiplier': float(np.mean(idx_est)) if idx_est else np.nan,
            'env_multiplier': float(np.mean(env_est)) if env_est else np.nan,
        })

    pres_df = pd.DataFrame(rows_presence)
    cnt_df = pd.DataFrame(rows_counts)

    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    # Presence panel
    x = np.arange(len(pres_df))
    w = 0.35
    axes[0].bar(x - w/2, pres_df['indices_multiplier'] - 1.0, w, label='Indices', color='#3498db')
    axes[0].bar(x + w/2, pres_df['env_multiplier'] - 1.0, w, label='Environmental', color='#e67e22')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(pres_df['response'], rotation=30, ha='right')
    axes[0].set_ylabel('Odds multiplier − 1 (per +1 SD)')
    axes[0].set_title('Presence Models (normalized by predictor SD)')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # Counts panel
    x2 = np.arange(len(cnt_df))
    axes[1].bar(x2 - w/2, cnt_df['indices_multiplier'] - 1.0, w, label='Indices', color='#3498db')
    axes[1].bar(x2 + w/2, cnt_df['env_multiplier'] - 1.0, w, label='Environmental', color='#e67e22')
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(cnt_df['response'], rotation=30, ha='right')
    axes[1].set_ylabel('Rate multiplier − 1 (per +1 SD)')
    axes[1].set_title('Count Models (normalized by predictor SD)')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fname = f"07_normalized_effects_panels_{'universal' if approach=='universal' else 'taxa'}.png"
    out_path = paths.get_figure_path(fname)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    return out_path


def _index_terms_from_coefs(df):
    meta_terms = set(['(Intercept)'] + ENV_COVARS)
    def is_index_term(t):
        if t in meta_terms:
            return False
        if str(t).startswith('month') or str(t).startswith('hour'):
            return False
        return True
    return [t for t in df['term'].unique() if is_index_term(t)]


def create_top_index_effects_panels(paths, approach='taxa', top_n=3):
    coefs = load_universal_coefficients(paths)
    coefs = coefs[coefs['effect'] == 'fixed'] if 'effect' in coefs.columns else coefs
    if approach == 'universal' and 'approach' in coefs.columns:
        coefs = coefs[coefs['approach'] == 'universal']
    elif approach == 'taxa' and 'approach' in coefs.columns:
        coefs = coefs[coefs['approach'] != 'universal']

    sdmap = _predictor_sd_map(paths, approach)

    def sd_for_term(term, response):
        if approach == 'universal':
            return sdmap.get(term, 1.0)
        group = 'fish' if response.startswith('fish_') else ('dolphin' if response.startswith('dolphin_') else 'vessel')
        return sdmap.get(group, {}).get(term, 1.0)

    def build_panel(responses, title, outfile):
        n = len(responses)
        fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(12, 3.5 * n))
        if n == 1:
            axes = [axes]
        for ax, resp in zip(axes, responses):
            sub = coefs[coefs['response'] == resp]
            idx_terms = _index_terms_from_coefs(sub)
            if not idx_terms:
                ax.set_visible(False)
                continue
            rows = []
            for term in idx_terms:
                row = sub[sub['term'] == term]
                if len(row) == 0:
                    continue
                beta = float(row['estimate'].iloc[0])
                sdx = sd_for_term(term, resp)
                mult = np.exp(beta * sdx)
                lo = row['conf.low'].iloc[0] if 'conf.low' in row.columns else np.nan
                hi = row['conf.high'].iloc[0] if 'conf.high' in row.columns else np.nan
                mlo = np.exp(float(lo) * sdx) if not pd.isna(lo) else np.nan
                mhi = np.exp(float(hi) * sdx) if not pd.isna(hi) else np.nan
                strength = abs(beta * sdx)
                rows.append({'term': term, 'mult': mult, 'mlo': mlo, 'mhi': mhi, 'strength': strength})
            df = pd.DataFrame(rows).sort_values('strength', ascending=False).head(top_n)
            ax.barh(df['term'], df['mult'] - 1.0, color='#2c3e50')
            for i, (_, r) in enumerate(df.iterrows()):
                if not pd.isna(r['mlo']) and not pd.isna(r['mhi']):
                    ax.errorbar(x=r['mult'] - 1.0, y=i, xerr=[[r['mult'] - r['mlo']], [r['mhi'] - r['mult']]], fmt='none', ecolor='#7f8c8d', capsize=3)
            ax.axvline(0, color='black', linewidth=1)
            ax.set_title(resp)
            ax.set_xlabel('Multiplier − 1 (per +1 SD)')
            ax.grid(axis='x', alpha=0.3)
        fig.suptitle(title)
        plt.tight_layout()
        out_path = paths.get_figure_path(outfile)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        return out_path

    p_out = build_panel(PRESENCE_RESPONSES, f'Top {top_n} Indices (Presence, {approach})', f'07_top_indices_presence_{approach}.png')
    c_out = build_panel(COUNT_RESPONSES, f'Top {top_n} Indices (Counts, {approach})', f'07_top_indices_counts_{approach}.png')
    return p_out, c_out


def create_delta_aic_panel(paths):
    stats_path = paths.processed_data / 'glmm_results' / 'glmm_model_stats_both_approaches.csv'
    df = pd.read_csv(stats_path)
    rows = []
    for resp in df['response'].unique():
        aic_univ = df[(df['response'] == resp) & (df['approach'] == 'universal')]['AIC']
        if len(aic_univ) == 0:
            continue
        aic_univ = float(aic_univ.iloc[0])
        aic_taxa = df[(df['response'] == resp) & (df['approach'].isin(['fish','dolphin','vessel']))]['AIC']
        if len(aic_taxa) == 0:
            continue
        best_taxa = float(aic_taxa.min())
        delta = aic_univ - best_taxa
        rows.append({'response': resp, 'delta_AIC': delta})
    plot_df = pd.DataFrame(rows).sort_values('delta_AIC', ascending=False)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(plot_df['response'], plot_df['delta_AIC'], color='#34495e')
    ax.axhline(0, color='black', linewidth=1)
    ax.set_ylabel('ΔAIC (Universal − Best Taxa-specific)')
    ax.set_title('Model Fit Improvement by Taxa-specific (higher = better than universal)')
    ax.tick_params(axis='x', rotation=30)
    ax.grid(axis='y', alpha=0.3)
    out_path = paths.get_figure_path('07_delta_aic_taxa_vs_universal.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    return out_path


def create_ablation_comparison_panels(paths):
    def make_panel(csv_path, title, outfile):
        df = pd.read_csv(csv_path)
        df = df[df['model'].isin(['env_station', 'idx_station', 'full'])]
        # Compute ΔAIC vs env_station per response
        base = df[df['model'] == 'env_station'][['response', 'AIC']].rename(columns={'AIC': 'AIC_env'})
        merged = df.merge(base, on='response', how='left')
        merged['delta_AIC'] = merged['AIC_env'] - merged['AIC']
        # Pivot for plotting
        plot_df = merged.pivot_table(index='response', columns='model', values='delta_AIC', aggfunc='mean')
        # Order models
        order = ['idx_station', 'full']
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(plot_df))
        w = 0.35
        ax.bar(x - w/2, plot_df[order[0]], w, label='Indices-only (+station)')
        ax.bar(x + w/2, plot_df[order[1]], w, label='Full (indices + env + station)')
        ax.axhline(0, color='black', linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df.index, rotation=30, ha='right')
        ax.set_ylabel('ΔAIC vs Env-only (+station)')
        ax.set_title(title)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        out_path = paths.get_figure_path(outfile)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        return out_path

    taxa_csvs = {
        'fish': paths.processed_data / 'glmm_results' / 'component_ablation_fish.csv',
        'dolphin': paths.processed_data / 'glmm_results' / 'component_ablation_dolphin.csv',
        'vessel': paths.processed_data / 'glmm_results' / 'component_ablation_vessel.csv',
    }
    univ_csv = paths.processed_data / 'glmm_results' / 'component_ablation_universal.csv'

    outs = []
    # Taxa-specific combined panel
    frames = []
    for g, p in taxa_csvs.items():
        if p.exists():
            frames.append(pd.read_csv(p))
    if frames:
        taxa_df = pd.concat(frames, ignore_index=True)
        tmp = paths.processed_data / 'glmm_results' / 'component_ablation_taxa_combined.csv'
        taxa_df.to_csv(tmp, index=False)
        outs.append(make_panel(tmp, 'Component Comparison (Taxa-specific)', '07_ablation_taxa_specific.png'))
    # Universal panel
    if univ_csv.exists():
        outs.append(make_panel(univ_csv, 'Component Comparison (Universal)', '07_ablation_universal.png'))
    return outs


def create_presence_top3_panels_taxa(paths, top_n=3):
    coefs = load_universal_coefficients(paths)
    coefs = coefs[coefs['effect'] == 'fixed'] if 'effect' in coefs.columns else coefs
    if 'approach' in coefs.columns:
        coefs = coefs[coefs['approach'] != 'universal']
    presence = coefs[coefs['response'].isin(PRESENCE_RESPONSES)]
    sdmap = _predictor_sd_map(paths, approach='taxa')

    rows = []
    for resp in PRESENCE_RESPONSES:
        sub = presence[presence['response'] == resp]
        idx_terms = _index_terms_from_coefs(sub)
        # rank by |beta|*SD
        scored = []
        for t in idx_terms:
            r = sub[sub['term'] == t]
            if len(r):
                beta = float(r['estimate'].iloc[0])
                sdx = _predictor_sd_map(paths, 'taxa').get('fish' if resp.startswith('fish_') else ('dolphin' if resp.startswith('dolphin_') else 'vessel'), {}).get(t, 1.0)
                scored.append((t, abs(beta * sdx)))
        scored.sort(key=lambda x: x[1], reverse=True)
        top = [t for t, _ in scored[:top_n]]
        for t in top:
            r = sub[sub['term'] == t]
            beta = float(r['estimate'].iloc[0])
            group = 'fish' if resp.startswith('fish_') else ('dolphin' if resp.startswith('dolphin_') else 'vessel')
            sdx = sdmap.get(group, {}).get(t, 1.0)
            mult = np.exp(beta * sdx)
            rows.append({'response': resp, 'index': t, 'multiplier_minus_1': mult - 1.0})

    plot_df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, resp in zip(axes, PRESENCE_RESPONSES):
        sub = plot_df[plot_df['response'] == resp]
        ax.bar(sub['index'], sub['multiplier_minus_1'], color='#2c3e50')
        ax.axhline(0, color='black', linewidth=1)
        ax.set_title(resp)
        ax.set_xticklabels(sub['index'], rotation=30, ha='right')
    axes[0].set_ylabel('Odds multiplier − 1 (per +1 SD)')
    fig.suptitle(f'Presence Models: Top {top_n} Taxa-specific Indices (normalized)')
    out_path = paths.get_figure_path('07_presence_top3_taxa.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    return out_path


def _load_ablation_df(paths, scope='taxa'):
    if scope == 'taxa':
        frames = []
        for name in ['component_ablation_fish.csv', 'component_ablation_dolphin.csv', 'component_ablation_vessel.csv']:
            p = paths.processed_data / 'glmm_results' / name
            if p.exists():
                frames.append(pd.read_csv(p))
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)
    else:
        p = paths.processed_data / 'glmm_results' / 'component_ablation_universal.csv'
        return pd.read_csv(p) if p.exists() else pd.DataFrame()


def create_ablation_slope_panels(paths):
    def slope_for_scope(scope, outfile, title):
        df = _load_ablation_df(paths, scope)
        if df.empty:
            return paths.get_figure_path(outfile)
        base = df[df['model'] == 'env_station'][['response', 'AIC']].rename(columns={'AIC': 'AIC_env'})
        merged = df.merge(base, on='response', how='left')
        pivot = merged[merged['model'].isin(['env_station', 'idx_station', 'full'])]
        fig, ax = plt.subplots(figsize=(12, 6))
        for resp in sorted(pivot['response'].unique()):
            sub = pivot[pivot['response'] == resp]
            a_env = float(sub[sub['model'] == 'env_station']['AIC_env'].iloc[0])
            a_idx = float(sub[sub['model'] == 'idx_station']['AIC'].iloc[0]) if ('idx_station' in sub['model'].values) else np.nan
            a_full = float(sub[sub['model'] == 'full']['AIC'].iloc[0]) if ('full' in sub['model'].values) else np.nan
            x = np.array([0, 1, 2])
            y = np.array([0, a_env - a_idx if not np.isnan(a_idx) else np.nan, a_env - a_full if not np.isnan(a_full) else np.nan])
            ax.plot(x, y, marker='o', alpha=0.4)
        ax.axhline(0, color='black', linewidth=1)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['Env-only', 'Indices-only', 'Full'])
        ax.set_ylabel('ΔAIC vs Env-only')
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)
        out_path = paths.get_figure_path(outfile)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        return out_path
    t = slope_for_scope('taxa', '07_ablation_slope_taxa.png', 'Ablation Slope (Taxa-specific)')
    u = slope_for_scope('universal', '07_ablation_slope_universal.png', 'Ablation Slope (Universal)')
    return t, u


def create_ablation_summary_bars(paths):
    def summary(scope):
        df = _load_ablation_df(paths, scope)
        if df.empty:
            return None
        base = df[df['model'] == 'env_station'][['response', 'AIC']].rename(columns={'AIC': 'AIC_env'})
        merged = df.merge(base, on='response', how='left')
        idx = merged[merged['model'] == 'idx_station']
        full = merged[merged['model'] == 'full']
        idx_delta = (idx['AIC_env'] - idx['AIC']).dropna().values
        full_delta = (full['AIC_env'] - full['AIC']).dropna().values
        return {
            'idx_mean': float(np.mean(idx_delta)) if len(idx_delta) else np.nan,
            'idx_std': float(np.std(idx_delta)) if len(idx_delta) else np.nan,
            'full_mean': float(np.mean(full_delta)) if len(full_delta) else np.nan,
            'full_std': float(np.std(full_delta)) if len(full_delta) else np.nan,
        }
    taxa = summary('taxa')
    univ = summary('universal')
    labels = ['Indices-only (Taxa)', 'Full (Taxa)', 'Indices-only (Universal)', 'Full (Universal)']
    means = [taxa['idx_mean'] if taxa else np.nan, taxa['full_mean'] if taxa else np.nan, univ['idx_mean'] if univ else np.nan, univ['full_mean'] if univ else np.nan]
    stds = [taxa['idx_std'] if taxa else np.nan, taxa['full_std'] if taxa else np.nan, univ['idx_std'] if univ else np.nan, univ['full_std'] if univ else np.nan]
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=4, color=['#2980b9', '#27ae60', '#8e44ad', '#f39c12'])
    ax.axhline(0, color='black', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylabel('Mean ΔAIC vs Env-only (±1 SD)')
    ax.set_title('Approach Comparison Summary')
    ax.grid(axis='y', alpha=0.3)
    out_path = paths.get_figure_path('07_ablation_summary.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    return out_path


def create_ablation_percentage_panels(paths):
    def pct_panel(csv_path, title, outfile):
        df = pd.read_csv(csv_path)
        df = df[df['model'].isin(['env_station', 'idx_station', 'full'])]
        base = df[df['model'] == 'env_station'][['response', 'AIC']].rename(columns={'AIC': 'AIC_env'})
        merged = df.merge(base, on='response', how='left')
        merged['delta_AIC'] = merged['AIC_env'] - merged['AIC']
        merged['pct_change'] = (merged['delta_AIC'] / merged['AIC_env']) * 100.0
        pivot = merged.pivot_table(index='response', columns='model', values='pct_change', aggfunc='mean')
        order = ['idx_station', 'full']
        # Sort by full improvement if available
        if 'full' in pivot.columns:
            pivot = pivot.sort_values('full', ascending=False)
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(pivot))
        w = 0.35
        b1 = ax.bar(x - w/2, pivot[order[0]], w, label='Indices-only (+station)', color='#3498db')
        b2 = ax.bar(x + w/2, pivot[order[1]], w, label='Full (indices + env + station)', color='#27ae60')
        ax.axhline(0, color='black', linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=30, ha='right')
        ax.set_ylabel('% change in AIC vs Env-only')
        ax.set_title(title)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        for b in list(b1) + list(b2):
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h + (1 if h >= 0 else -1), f'{h:.1f}%', ha='center', va='bottom', fontsize=8)
        out_path = paths.get_figure_path(outfile)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        return out_path

    taxa_csvs = [paths.processed_data / 'glmm_results' / 'component_ablation_fish.csv',
                 paths.processed_data / 'glmm_results' / 'component_ablation_dolphin.csv',
                 paths.processed_data / 'glmm_results' / 'component_ablation_vessel.csv']
    frames = [pd.read_csv(p) for p in taxa_csvs if p.exists()]
    outs = []
    if frames:
        taxa_df = pd.concat(frames, ignore_index=True)
        tmp_taxa = paths.processed_data / 'glmm_results' / 'component_ablation_taxa_combined.csv'
        taxa_df.to_csv(tmp_taxa, index=False)
        outs.append(pct_panel(tmp_taxa, 'AIC % Change vs Env-only (Taxa-specific)', '07_ablation_pct_taxa.png'))

    univ_csv = paths.processed_data / 'glmm_results' / 'component_ablation_universal.csv'
    if univ_csv.exists():
        outs.append(pct_panel(univ_csv, 'AIC % Change vs Env-only (Universal)', '07_ablation_pct_universal.png'))
    return outs


def create_index_reduction_flow(paths):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')
    ax.text(0.1, 0.5, '1) Correlation pruning\nRemove near-duplicates', bbox=dict(boxstyle='round', fc='#ecf0f1'), fontsize=11)
    ax.text(0.42, 0.5, '2) Random-forest screening\nFind informative indices', bbox=dict(boxstyle='round', fc='#dfe6e9'), fontsize=11)
    ax.text(0.74, 0.5, '3) VIF selection\nCompact, non-redundant set', bbox=dict(boxstyle='round', fc='#ecf0f1'), fontsize=11)
    ax.annotate('', xy=(0.38, 0.52), xytext=(0.28, 0.52), arrowprops=dict(arrowstyle='->'))
    ax.annotate('', xy=(0.70, 0.52), xytext=(0.60, 0.52), arrowprops=dict(arrowstyle='->'))
    out_path = paths.get_figure_path('07_index_reduction_flow.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    return out_path


def create_env_vs_full_panels(paths):
    def split_presence_counts(df):
        presence = ['fish_present', 'dolphin_present', 'vessel_present']
        counts = ['fish_richness', 'fish_activity', 'dolphin_activity', 'dolphin_whistles', 'dolphin_burst_pulses', 'dolphin_echolocation']
        df_p = df[df['response'].isin(presence)].copy()
        df_c = df[df['response'].isin(counts)].copy()
        return df_p, df_c

    # Load ablation data (taxa-specific preferred, fall back to universal)
    df_taxa = _load_ablation_df(paths, scope='taxa')
    df_univ = _load_ablation_df(paths, scope='universal')
    if df_taxa.empty and df_univ.empty:
        return (
            paths.get_figure_path('07_env_vs_full_presence.png'),
            paths.get_figure_path('07_env_vs_full_counts.png'),
            paths.get_figure_path('07_env_vs_full_presence_pct.png'),
            paths.get_figure_path('07_env_vs_full_counts_pct.png'),
            paths.get_figure_path('07_env_vs_full_all_pct.png'),
        )

    frames = []
    if not df_taxa.empty:
        df_taxa['src_order'] = 0
        frames.append(df_taxa)
    if not df_univ.empty:
        df_univ['src_order'] = 1
        frames.append(df_univ)
    all_df = pd.concat(frames, ignore_index=True)
    # Prefer taxa-specific rows, then universal
    all_df = all_df.sort_values(['response', 'model', 'src_order']).drop_duplicates(['response', 'model'], keep='first')

    base = all_df[all_df['model'] == 'env_station'][['response', 'AIC']].rename(columns={'AIC': 'AIC_env'})
    full = all_df[all_df['model'] == 'full'][['response', 'AIC']].rename(columns={'AIC': 'AIC_full'})
    merged = base.merge(full, on='response', how='left')
    merged['delta_AIC'] = merged['AIC_env'] - merged['AIC_full']
    merged['pct_change'] = (merged['delta_AIC'] / merged['AIC_env']) * 100.0

    df_p, df_c = split_presence_counts(merged)

    # Paper ΔAIC panels
    def plot_delta(panel_df, title, outfile):
        panel_df = panel_df.sort_values('delta_AIC', ascending=False)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(panel_df['response'], panel_df['delta_AIC'], color='#34495e')
        ax.axhline(0, color='black', linewidth=1)
        ax.set_ylabel('ΔAIC (Env-only − Full)')
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=30)
        ax.grid(axis='y', alpha=0.3)
        out_path = paths.get_figure_path(outfile)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        return out_path

    p_out = plot_delta(df_p, 'Env vs Full (Presence, ΔAIC)', '07_env_vs_full_presence.png')
    c_out = plot_delta(df_c, 'Env vs Full (Counts, ΔAIC)', '07_env_vs_full_counts.png')

    # Slides % panels
    def plot_pct(panel_df, title, outfile):
        panel_df = panel_df.sort_values('pct_change', ascending=False)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(panel_df['response'], panel_df['pct_change'], color='#2c3e50')
        ax.axhline(0, color='black', linewidth=1)
        ax.set_ylabel('% change in AIC vs Env-only')
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=30)
        ax.grid(axis='y', alpha=0.3)
        out_path = paths.get_figure_path(outfile)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        return out_path

    p_pct = plot_pct(df_p, 'Env vs Full (Presence, %ΔAIC)', '07_env_vs_full_presence_pct.png')
    c_pct = plot_pct(df_c, 'Env vs Full (Counts, %ΔAIC)', '07_env_vs_full_counts_pct.png')

    d_all = pd.concat([
        df_p.assign(group='Presence'),
        df_c.assign(group='Counts')
    ], ignore_index=True)
    d_all = d_all.sort_values('pct_change', ascending=False)
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(d_all['response'], d_all['pct_change'], color='#2c3e50')
    ax.axhline(0, color='black', linewidth=1)
    ax.set_ylabel('% change in AIC vs Env-only')
    ax.set_title('Env vs Full (%ΔAIC) — Presence and Counts (within-response)')
    ax.tick_params(axis='x', rotation=30)
    ax.grid(axis='y', alpha=0.3)
    out_all = paths.get_figure_path('07_env_vs_full_all_pct.png')
    plt.tight_layout()
    plt.savefig(out_all, dpi=300, bbox_inches='tight')
    plt.close()

    return p_out, c_out, p_pct, c_pct, out_all


def create_presence_top3_panels_taxa_wide(paths, top_n=3):
    coefs = load_universal_coefficients(paths)
    coefs = coefs[coefs['effect'] == 'fixed'] if 'effect' in coefs.columns else coefs
    if 'approach' in coefs.columns:
        coefs = coefs[coefs['approach'] != 'universal']
    sdmap = _predictor_sd_map(paths, approach='taxa')
    rows = []
    for resp in PRESENCE_RESPONSES:
        sub = coefs[coefs['response'] == resp]
        idx_terms = _index_terms_from_coefs(sub)
        scored = []
        for t in idx_terms:
            r = sub[sub['term'] == t]
            if len(r):
                beta = float(r['estimate'].iloc[0])
                group = 'fish' if resp.startswith('fish_') else ('dolphin' if resp.startswith('dolphin_') else 'vessel')
                sdx = sdmap.get(group, {}).get(t, 1.0)
                scored.append((t, abs(beta * sdx), float(np.exp(beta * sdx) - 1.0)))
        scored.sort(key=lambda x: x[1], reverse=True)
        for t, _, multm1 in scored[:top_n]:
            rows.append({'response': resp, 'index': t, 'multiplier_minus_1': multm1})
    plot_df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(16, 9), sharey=True)
    for ax, resp in zip(axes, PRESENCE_RESPONSES):
        sub = plot_df[plot_df['response'] == resp]
        ax.bar(sub['index'], sub['multiplier_minus_1'], color='#2c3e50')
        ax.axhline(0, color='black', linewidth=1)
        ax.set_title(resp, fontsize=14)
        ax.tick_params(axis='x', rotation=30)
        ax.grid(axis='y', alpha=0.3)
    axes[0].set_ylabel('Odds multiplier − 1 (per +1 SD)')
    out_path = paths.get_figure_path('07_top_indices_presence_taxa_wide.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    return out_path

def main():
    paths = ProjectPaths()
    paths.ensure_output_dirs()
    h1 = create_coefficient_heatmap(paths)
    h2 = plot_marginal_effects(paths)
    h3 = create_magnitude_bars(paths)
    h4 = create_odds_ratio_bars(paths)
    h5 = create_delta_probability_bars(paths)
    n_univ = create_normalized_effect_panels(paths, approach='universal')
    n_taxa = create_normalized_effect_panels(paths, approach='taxa')
    t_presence_univ, t_counts_univ = create_top_index_effects_panels(paths, approach='universal', top_n=3)
    t_presence_taxa, t_counts_taxa = create_top_index_effects_panels(paths, approach='taxa', top_n=3)
    d_aic = create_delta_aic_panel(paths)
    ablation_outs = create_ablation_comparison_panels(paths)
    p_top3_taxa = create_presence_top3_panels_taxa(paths, top_n=3)
    slope_taxa, slope_univ = create_ablation_slope_panels(paths)
    ablation_summary = create_ablation_summary_bars(paths)
    perc_outs = create_ablation_percentage_panels(paths)
    env_full_presence, env_full_counts, env_full_presence_pct, env_full_counts_pct, env_full_combined_pct = create_env_vs_full_panels(paths)
    presence_wide = create_presence_top3_panels_taxa_wide(paths, top_n=3)
    counts_grid = create_top_indices_counts_grid_taxa(paths, top_n=3)
    print('Figures saved:')
    print(f'  - {h1}')
    print(f'  - {h2}')
    print(f'  - {h3}')
    print(f'  - {h4}')
    print(f'  - {h5}')
    print(f'  - {n_univ}')
    print(f'  - {n_taxa}')
    print(f'  - {t_presence_univ}')
    print(f'  - {t_counts_univ}')
    print(f'  - {t_presence_taxa}')
    print(f'  - {t_counts_taxa}')
    print(f'  - {d_aic}')
    for o in ablation_outs:
        print(f'  - {o}')
    print(f'  - {p_top3_taxa}')
    print(f'  - {slope_taxa}')
    print(f'  - {slope_univ}')
    print(f'  - {ablation_summary}')
    for o in perc_outs:
        print(f'  - {o}')
    print(f'  - {env_full_presence}')
    print(f'  - {env_full_counts}')
    print(f'  - {env_full_presence_pct}')
    print(f'  - {env_full_counts_pct}')
    print(f'  - {env_full_combined_pct}')
    print(f'  - {presence_wide}')
    print(f'  - {counts_grid}')

def create_top_indices_counts_grid_taxa(paths, top_n=3):
    coefs = load_universal_coefficients(paths)
    coefs = coefs[coefs['effect'] == 'fixed'] if 'effect' in coefs.columns else coefs
    if 'approach' in coefs.columns:
        coefs = coefs[coefs['approach'] != 'universal']
    sdmap = _predictor_sd_map(paths, approach='taxa')
    responses = ['fish_richness', 'fish_activity', 'dolphin_activity', 'dolphin_whistles', 'dolphin_burst_pulses', 'dolphin_echolocation']
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    for i, resp in enumerate(responses):
        sub = coefs[coefs['response'] == resp]
        idx_terms = _index_terms_from_coefs(sub)
        rows = []
        for t in idx_terms:
            r = sub[sub['term'] == t]
            if len(r):
                beta = float(r['estimate'].iloc[0])
                group = 'fish' if resp.startswith('fish_') else 'dolphin'
                sdx = sdmap.get(group, {}).get(t, 1.0)
                rows.append({'term': t, 'strength': abs(beta * sdx), 'mult': float(np.exp(beta * sdx) - 1.0)})
        df = pd.DataFrame(rows).sort_values('strength', ascending=False).head(top_n)
        ax = axes[i]
        ax.bar(df['term'], df['mult'], color='#2c3e50')
        ax.axhline(0, color='black', linewidth=1)
        ax.set_title(resp, fontsize=12)
        ax.tick_params(axis='x', rotation=30)
        ax.grid(axis='y', alpha=0.3)
    axes[0].set_ylabel('Rate multiplier − 1 (per +1 SD)')
    out_path = paths.get_figure_path('07_top_indices_counts_taxa_grid.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    return out_path


if __name__ == '__main__':
    main()