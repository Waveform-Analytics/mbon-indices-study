"""
Stage 04: Exploratory Visualization

Generates descriptive summaries and visualizations for sanity checking
before modeling:
- Descriptive stats table (by station/season/hour)
- Distribution plots for community metrics
- Scatter overlays: community metrics vs acoustic indices (with Pearson r)
- Heatmaps: per station showing values by date × hour (local time)
"""

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

root = Path(__file__).parent.parent
sys.path.append(str(root / "src" / "python"))

from mbon_indices.config import load_analysis_config
from mbon_indices.data import load_processed_parquet, load_final_indices_list
from mbon_indices.utils.logging import setup_stage_logging


def get_season(month: int) -> str:
    """Map month number to meteorological season."""
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Fall"


def generate_descriptive_stats(
    df: pd.DataFrame,
    variables: list[str],
    output_path: Path
) -> pd.DataFrame:
    """
    Generate descriptive statistics table by station, season, and hour.

    Parameters:
        df: Analysis-ready DataFrame
        variables: List of column names to summarize
        output_path: Path to save CSV output

    Returns:
        DataFrame with descriptive statistics
    """
    # Add season column based on month from datetime_local
    df = df.copy()
    df['season'] = df['datetime_local'].dt.month.apply(get_season)

    # Filter to variables that exist in the dataframe
    vars_present = [v for v in variables if v in df.columns]
    vars_missing = [v for v in variables if v not in df.columns]
    if vars_missing:
        print(f"  Warning: Variables not found in data: {vars_missing}")

    # Compute stats grouped by station, season, hour
    stats_rows = []

    # Get actual hours present in data (may be odd or even depending on timezone offset)
    hours_in_data = sorted(df['hour_of_day'].unique())

    for station in sorted(df['station'].unique()):
        for season in ['Winter', 'Spring', 'Summer', 'Fall']:
            for hour in hours_in_data:
                subset = df[
                    (df['station'] == station) &
                    (df['season'] == season) &
                    (df['hour_of_day'] == hour)
                ]

                if len(subset) == 0:
                    continue

                row = {
                    'station': station,
                    'season': season,
                    'hour': hour,
                    'n_obs': len(subset)
                }

                for var in vars_present:
                    row[f'{var}_mean'] = subset[var].mean()
                    row[f'{var}_sd'] = subset[var].std()
                    row[f'{var}_min'] = subset[var].min()
                    row[f'{var}_max'] = subset[var].max()

                stats_rows.append(row)

    stats_df = pd.DataFrame(stats_rows)

    # Save to CSV
    stats_df.to_csv(output_path, index=False)
    print(f"✓ Saved descriptive stats: {output_path}")
    print(f"    {len(stats_df)} rows (station × season × hour combinations)")
    print(f"    {len(vars_present)} variables summarized")

    return stats_df


def plot_distributions(
    df: pd.DataFrame,
    variables: list[str],
    output_path: Path,
    figsize: tuple[int, int] = (14, 10)
) -> None:
    """
    Create faceted histogram figure for community metrics.

    Parameters:
        df: Analysis-ready DataFrame
        variables: List of column names to plot
        output_path: Path to save figure
        figsize: Figure size in inches
    """
    # Filter to variables that exist
    vars_present = [v for v in variables if v in df.columns]
    n_vars = len(vars_present)

    if n_vars == 0:
        print("  Warning: No variables found for distribution plots")
        return

    # Determine grid layout (aim for roughly square)
    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols  # Ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()  # Make it easy to iterate

    for i, var in enumerate(vars_present):
        ax = axes[i]
        data = df[var].dropna()

        # Choose appropriate bins based on data type
        if data.nunique() <= 10:
            # Discrete/binary data - use exact values
            bins = range(int(data.min()), int(data.max()) + 2)
            ax.hist(data, bins=bins, edgecolor='black', alpha=0.7, align='left')
        else:
            # Continuous data - use auto bins
            ax.hist(data, bins=30, edgecolor='black', alpha=0.7)

        ax.set_xlabel(var)
        ax.set_ylabel('Count')
        ax.set_title(f'{var}\n(n={len(data):,}, mean={data.mean():.2f})')

    # Hide unused subplots
    for i in range(n_vars, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('Community Metrics Distributions', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved distribution plots: {output_path}")
    print(f"    {n_vars} variables plotted")


def is_binary(series: pd.Series) -> bool:
    """Check if a series contains only binary values (0 and 1)."""
    unique_vals = series.dropna().unique()
    return set(unique_vals).issubset({0, 1, 0.0, 1.0})


def is_skewed(series: pd.Series, threshold: float = 2.0) -> bool:
    """Check if a series is highly skewed (would benefit from log scale)."""
    from scipy.stats import skew
    data = series.dropna()
    if len(data) == 0:
        return False
    return abs(skew(data)) > threshold


def clear_directory(dir_path: Path) -> None:
    """Remove all files in a directory."""
    if dir_path.exists():
        for f in dir_path.glob('*'):
            if f.is_file():
                f.unlink()


def plot_indices_vs_response(
    df: pd.DataFrame,
    responses: list[str],
    indices: list[str],
    output_dir: Path,
    panel_size: float = 3.5,
    log_scale: bool = False
) -> None:
    """
    Create scatter/violin plots of community metrics vs acoustic indices with Pearson r.

    Creates one figure per response variable, with subplots for each index.
    Uses violin plots for binary responses, scatter plots for continuous.
    Grid layout adapts dynamically to the number of indices.

    Parameters:
        df: Analysis-ready DataFrame
        responses: List of response variable column names
        indices: List of acoustic index column names to plot against
        output_dir: Directory to save figures
        panel_size: Size of each panel in inches (panels are square)
        log_scale: If True, use log(y+1) scale for y-axis
    """
    import math
    from scipy import stats

    # Filter to indices that exist in the data
    indices_present = [idx for idx in indices if idx in df.columns]
    indices_missing = [idx for idx in indices if idx not in df.columns]
    if indices_missing:
        print(f"  Warning: Some indices not found in data: {indices_missing}")
        print(f"           Available indices in data/processed/indices_final.csv")

    if len(indices_present) == 0:
        print("  Error: No valid indices found for plots, skipping")
        return

    # Dynamic grid layout (roughly square)
    n_indices = len(indices_present)
    n_cols = math.ceil(math.sqrt(n_indices))
    n_rows = math.ceil(n_indices / n_cols)

    # Calculate figure size for square panels
    figsize = (n_cols * panel_size, n_rows * panel_size)

    suffix = '_log' if log_scale else ''

    for response in responses:
        if response not in df.columns:
            print(f"  Warning: Response '{response}' not found in data, skipping")
            continue

        # Check if response is binary
        response_is_binary = is_binary(df[response])

        # Skip log scale for binary responses (doesn't make sense)
        if response_is_binary and log_scale:
            continue

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_indices == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i, idx in enumerate(indices_present):
            ax = axes[i]

            # Get data, dropping NaN
            mask = df[response].notna() & df[idx].notna()
            x = df.loc[mask, idx]
            y = df.loc[mask, response]

            # Apply log transform if requested
            if log_scale:
                y_plot = np.log1p(y)  # log(y + 1)
                y_label = f'log({response} + 1)'
            else:
                y_plot = y
                y_label = response

            # Compute Pearson r (on transformed data if log scale)
            r, p = stats.pearsonr(x, y_plot)
            sig_stars = ' ***' if p < 0.001 else ' **' if p < 0.01 else ' *' if p < 0.05 else ''

            if response_is_binary:
                # Violin plot for binary response
                plot_data = pd.DataFrame({'index': x, 'response': y.astype(int)})
                parts = ax.violinplot(
                    [plot_data[plot_data['response'] == 0]['index'].values,
                     plot_data[plot_data['response'] == 1]['index'].values],
                    positions=[0, 1],
                    showmeans=True,
                    showmedians=True
                )
                # Color the violins
                for pc in parts['bodies']:
                    pc.set_facecolor('steelblue')
                    pc.set_alpha(0.7)
                ax.set_xticks([0, 1])
                ax.set_xticklabels(['Absent (0)', 'Present (1)'])
                ax.set_ylabel(idx)
                ax.set_xlabel(response)
            else:
                # Scatter plot for continuous response
                ax.scatter(x, y_plot, alpha=0.3, s=10, edgecolors='none')

                # Add regression line
                z = np.polyfit(x, y_plot, 1)
                p_line = np.poly1d(z)
                x_line = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_line, p_line(x_line), 'r-', linewidth=2, alpha=0.8)

                ax.set_xlabel(idx)
                ax.set_ylabel(y_label)

            ax.set_title(f'r = {r:.3f}{sig_stars}', fontsize=10)

        # Hide unused subplots
        for i in range(n_indices, len(axes)):
            axes[i].set_visible(False)

        title_suffix = ' (log scale)' if log_scale else ''
        plt.suptitle(f'{response} vs Acoustic Indices{title_suffix}', fontsize=14)
        plt.tight_layout()

        # Save figure
        output_path = output_dir / f'indices_vs_{response}{suffix}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()


def plot_scatter_overlays(
    df: pd.DataFrame,
    responses: list[str],
    indices: list[str],
    output_dir: Path,
    panel_size: float = 3.5
) -> None:
    """
    Create index vs response plots with automatic log scale detection.

    Clears output directory first, then generates:
    - Linear scale plots for all responses
    - Log scale plots for skewed non-binary responses
    """
    # Clear the output directory first
    clear_directory(output_dir)

    # Generate linear scale plots for all responses
    plot_indices_vs_response(df, responses, indices, output_dir, panel_size, log_scale=False)

    # Detect skewed responses and generate log scale plots
    skewed_responses = [r for r in responses if r in df.columns
                        and not is_binary(df[r]) and is_skewed(df[r])]

    if skewed_responses:
        print(f"  Generating log-scale plots for skewed responses: {skewed_responses}")
        plot_indices_vs_response(df, skewed_responses, indices, output_dir, panel_size, log_scale=True)

    # Count files generated
    n_files = len(list(output_dir.glob('*.png')))
    print(f"✓ Saved index vs response plots: {output_dir}")
    print(f"    {n_files} figures generated")


def plot_heatmaps(
    df: pd.DataFrame,
    variables: list[str],
    output_dir: Path,
    cmap: str = 'viridis',
    midnight_center: bool = True
) -> None:
    """
    Create date × hour heatmaps for each variable, with vertically stacked panels per station.

    Parameters:
        df: Analysis-ready DataFrame with datetime_local, hour_of_day, station columns
        variables: List of column names to create heatmaps for
        output_dir: Directory to save figures
        cmap: Colormap name for heatmaps
        midnight_center: If True, reorder hours so midnight is in center of y-axis
    """
    # Clear output directory
    clear_directory(output_dir)

    # Get stations
    stations = sorted(df['station'].unique())
    n_stations = len(stations)

    # Extract date from datetime_local
    df = df.copy()
    df['date'] = df['datetime_local'].dt.date

    # Filter to variables that exist
    vars_present = [v for v in variables if v in df.columns]
    if not vars_present:
        print("  Warning: No variables found for heatmaps")
        return

    for var in vars_present:
        # Vertically stacked layout: wide panels (16 inches) with modest height each (3 inches)
        # Extra height at bottom for colorbar
        fig_width = 16
        panel_height = 3
        fig_height = n_stations * panel_height + 1.5  # Extra space for colorbar and title

        fig, axes = plt.subplots(
            n_stations, 1,
            figsize=(fig_width, fig_height),
            sharex=True
        )
        if n_stations == 1:
            axes = [axes]

        # Track min/max for consistent color scale across stations
        vmin = df[var].min()
        vmax = df[var].max()

        for i, station in enumerate(stations):
            ax = axes[i]
            station_data = df[df['station'] == station]

            # Create pivot table: rows=hour, cols=date, values=variable
            pivot = station_data.pivot_table(
                index='hour_of_day',
                columns='date',
                values=var,
                aggfunc='mean'
            )

            # Reorder hours if midnight_center
            if midnight_center:
                # Shift so midnight (0 or 1 depending on offset) is in the middle
                hours_present = sorted(pivot.index)
                shift_point = len(hours_present) // 2
                new_order = hours_present[shift_point:] + hours_present[:shift_point]
                pivot = pivot.reindex(new_order)

            # Plot heatmap
            im = ax.imshow(
                pivot.values,
                aspect='auto',
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation='nearest'
            )

            # Y-axis: hours (on left side with station label)
            if midnight_center:
                y_labels = new_order
            else:
                y_labels = sorted(pivot.index)
            ax.set_yticks(range(len(y_labels)))
            ax.set_yticklabels([f'{h:02d}:00' for h in y_labels])
            ax.set_ylabel(f'{station}\nHour (local)', fontsize=10)

            # X-axis: only show on bottom panel
            n_dates = len(pivot.columns)
            if i == n_stations - 1:
                # Show ~12 date labels on bottom panel
                if n_dates > 12:
                    step = n_dates // 12
                    tick_positions = list(range(0, n_dates, step))
                    tick_labels = [pivot.columns[j].strftime('%b') if j < n_dates else '' for j in tick_positions]
                    ax.set_xticks(tick_positions)
                    ax.set_xticklabels(tick_labels)
                else:
                    ax.set_xticks(range(n_dates))
                    ax.set_xticklabels([d.strftime('%m-%d') for d in pivot.columns], rotation=45, ha='right')
                ax.set_xlabel('Date (2021)')
            else:
                ax.set_xticks([])

        # Add horizontal colorbar at bottom
        cbar_ax = fig.add_axes([0.15, 0.06, 0.7, 0.02])  # [left, bottom, width, height]
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar.set_label(var, fontsize=11)

        plt.suptitle(f'{var} by Date and Hour', fontsize=14, y=0.98)

        # Adjust layout to make room for colorbar
        plt.subplots_adjust(bottom=0.12, top=0.94, hspace=0.15)

        # Save
        output_path = output_dir / f'heatmap_{var}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"✓ Saved heatmaps: {output_dir}")
    print(f"    {len(vars_present)} variables × {n_stations} stations")


def load_data(root: Path) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    """
    Load analysis-ready data and identify column groups.

    Returns:
        df: Analysis-ready DataFrame
        responses: List of response variable column names (9 community metrics)
        indices: List of acoustic index column names (20 final indices)
        covariates: List of covariate column names (temperature, depth)
    """
    # Load analysis-ready dataset
    df = load_processed_parquet(root, "analysis_ready")
    print(f"✓ Loaded analysis-ready data: {len(df):,} rows, {len(df.columns)} columns")

    # Load config to identify column groups
    cfg = load_analysis_config(root)

    # Response variables (community metrics) - config has dict with family info
    responses_cfg = cfg.get('responses', {})
    responses = list(responses_cfg.keys())
    print(f"✓ Response variables: {len(responses)}")
    for r in responses:
        print(f"    - {r}")

    # Acoustic indices
    indices = load_final_indices_list(root)
    print(f"✓ Acoustic indices: {len(indices)}")

    # Covariates - config has dict with true/false flags
    covariates_cfg = cfg.get('covariates', {})
    covariates = [k for k, v in covariates_cfg.items() if v is True]
    print(f"✓ Covariates: {len(covariates)}")
    for c in covariates:
        print(f"    - {c}")

    return df, responses, indices, covariates


def main():
    # Set up logging
    logger = setup_stage_logging(root, "stage04_exploratory_viz")

    try:
        print("=" * 60)
        print("STAGE 04: EXPLORATORY VISUALIZATION")
        print("=" * 60)
        print()

        # Step 1: Load data and identify column groups
        print("Step 1: Loading data...")
        df, responses, indices, covariates = load_data(root)
        print()

        # Create output directories
        (root / "results" / "tables").mkdir(parents=True, exist_ok=True)
        (root / "results" / "figures" / "exploratory").mkdir(parents=True, exist_ok=True)

        # Step 2: Descriptive stats table
        print("Step 2: Generating descriptive stats...")
        stats_vars = responses + covariates  # Summarize community metrics and covariates
        stats_path = root / "results" / "tables" / "descriptive_stats.csv"
        generate_descriptive_stats(df, stats_vars, stats_path)
        print()

        # Step 3: Distribution plots
        print("Step 3: Creating distribution plots...")
        dist_path = root / "results" / "figures" / "exploratory" / "community_metrics_distributions.png"
        plot_distributions(df, responses, dist_path)
        print()

        # Step 4: Scatter overlays (all indices)
        print("Step 4: Creating scatter overlays...")
        scatter_dir = root / "results" / "figures" / "exploratory" / "scatter"
        scatter_dir.mkdir(parents=True, exist_ok=True)
        plot_scatter_overlays(df, responses, indices, scatter_dir)
        print()

        # Step 5: Heatmaps
        print("Step 5: Creating heatmaps...")
        heatmap_dir = root / "results" / "figures" / "exploratory" / "heatmaps"
        heatmap_dir.mkdir(parents=True, exist_ok=True)

        # Load config for heatmap settings
        cfg = load_analysis_config(root)
        exploratory_cfg = cfg.get('exploratory', {})
        cmap = exploratory_cfg.get('heatmap_color_scheme', 'viridis')
        midnight_center = exploratory_cfg.get('heatmap_midnight_center', True)

        # Heatmap variables: responses + all final indices + covariates
        heatmap_vars = responses + indices + covariates
        plot_heatmaps(df, heatmap_vars, heatmap_dir, cmap=cmap, midnight_center=midnight_center)
        print()

        print("=" * 60)
        print("✓ Stage 04 complete")
        print("=" * 60)
        print()
        print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    finally:
        # Close logger and restore stdout
        logger.close()
        sys.stdout = logger.terminal


if __name__ == "__main__":
    main()