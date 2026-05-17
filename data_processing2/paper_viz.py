import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns

PUBLICATION_STYLE = {
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.labelsize': 14,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'axes.linewidth': 1.2,
    'grid.alpha': 0.3,
    'figure.facecolor': 'white',
    'legend.fontsize': 12,
}


def plot_dst_timeseries(time_stamps, true_y, pred_y, title="Dst Predictions",
                        figsize=(16, 8), storm_thresholds=True, save_path=None, dpi=300):
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update(PUBLICATION_STYLE)

    time_stamps = np.array(time_stamps)
    true_y = np.array(true_y)
    pred_y = np.array(pred_y)

    if not (len(time_stamps) == len(true_y) == len(pred_y)):
        raise ValueError(f"Array size mismatch: timestamps={len(time_stamps)}, "
                         f"true_y={len(true_y)}, pred_y={len(pred_y)}.")

    if len(time_stamps) > 0 and isinstance(time_stamps[0], str):
        time_stamps = pd.to_datetime(time_stamps)
    elif not isinstance(time_stamps, pd.DatetimeIndex):
        time_stamps = pd.DatetimeIndex(time_stamps)

    rmse = np.sqrt(mean_squared_error(true_y, pred_y))
    mae = mean_absolute_error(true_y, pred_y)
    r2 = r2_score(true_y, pred_y)
    correlation = np.corrcoef(true_y, pred_y)[0, 1]

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(time_stamps, true_y, color='#2E86AB', linewidth=1.5, alpha=0.8,
            label='Observed Dst', linestyle='-')
    ax.plot(time_stamps, pred_y, color='#F24236', linewidth=1.2, alpha=0.9,
            label='Predicted Dst', linestyle='--')

    if storm_thresholds:
        ax.axhline(y=-50, color='orange', linestyle=':', linewidth=2, alpha=0.7,
                   label='Moderate Storm ($-$50 nT)')
        ax.axhline(y=-100, color='red', linestyle=':', linewidth=2, alpha=0.7,
                   label='Intense Storm ($-$100 nT)')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

    ax.set_ylabel('Dst Index (nT)')
    ax.set_xlabel('Date')

    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)

    metrics_text = f'RMSE: {rmse:.2f} nT\nMAE: {mae:.2f} nT\nR²: {r2:.3f}\nρ: {correlation:.3f}'
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    fig.autofmt_xdate()
    plt.tight_layout()

    if save_path:
        save_file = f"{save_path}_timeseries.png"
        plt.savefig(save_file, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_file}")

    return fig


def plot_dst_scatter(time_stamps, true_y, pred_y, title="Dst Predictions",
                     figsize=(10, 10), storm_thresholds=True, save_path=None, dpi=300,
                     hexbin_plot=False):
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update(PUBLICATION_STYLE)

    time_stamps = np.array(time_stamps)
    true_y = np.array(true_y)
    pred_y = np.array(pred_y)

    if not (len(time_stamps) == len(true_y) == len(pred_y)):
        raise ValueError(f"Array size mismatch: timestamps={len(time_stamps)}, "
                         f"true_y={len(true_y)}, pred_y={len(pred_y)}.")

    if len(time_stamps) > 0 and isinstance(time_stamps[0], str):
        time_stamps_dt = pd.to_datetime(time_stamps)
    elif not isinstance(time_stamps, pd.DatetimeIndex):
        time_stamps_dt = pd.DatetimeIndex(time_stamps)
    else:
        time_stamps_dt = time_stamps

    rmse = np.sqrt(mean_squared_error(true_y, pred_y))
    mae = mean_absolute_error(true_y, pred_y)
    r2 = r2_score(true_y, pred_y)
    correlation = np.corrcoef(true_y, pred_y)[0, 1]

    fig, ax = plt.subplots(figsize=figsize)

    if hexbin_plot:
        hb = ax.hexbin(true_y, pred_y, gridsize=25, cmap='Blues', alpha=1, mincnt=1)
        cb = plt.colorbar(hb, ax=ax, label='Point Density', shrink=0.8)
        cb.ax.tick_params(labelsize=12)
    else:
        ax.scatter(true_y, pred_y, alpha=0.6, s=25, color='#4CAF50',
                   edgecolors='white', linewidth=0.5)

    min_val = min(np.min(true_y), np.min(pred_y))
    max_val = max(np.max(true_y), np.max(pred_y))
    ax.plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=3, alpha=0.8,
            label='Perfect Prediction', zorder=10)

    slope, intercept, r_value, p_value, std_err = stats.linregress(true_y, pred_y)
    regression_line = slope * np.array([min_val, max_val]) + intercept
    ax.plot([min_val, max_val], regression_line, 'orange', linewidth=3, alpha=0.8,
            linestyle='--', label=f'Best Fit: y = {slope:.2f}x + {intercept:.1f}', zorder=9)

    if storm_thresholds:
        ax.axvline(x=-50, color='orange', linestyle=':', linewidth=2, alpha=0.6,
                   label='Moderate Storm Threshold')
        ax.axhline(y=-50, color='orange', linestyle=':', linewidth=2, alpha=0.6)
        ax.axvline(x=-100, color='red', linestyle=':', linewidth=2, alpha=0.6,
                   label='Intense Storm Threshold')
        ax.axhline(y=-100, color='red', linestyle=':', linewidth=2, alpha=0.6)

        ax.text(-75, max_val * 0.95, 'False\nNegatives', ha='center', va='top',
                fontsize=11, alpha=0.7, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
        ax.text(max_val * 0.95, -75, 'False\nPositives', ha='right', va='center',
                fontsize=11, alpha=0.7, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

    ax.set_xlabel('Observed Dst (nT)')
    ax.set_ylabel('Predicted Dst (nT)')

    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, which="major", alpha=0.3)
    ax.xaxis.set_major_locator(plt.MultipleLocator(50))
    ax.yaxis.set_major_locator(plt.MultipleLocator(50))

    metrics_text = (f'Performance Metrics:\n'
                    f'RMSE: {rmse:.2f} nT\n'
                    f'MAE: {mae:.2f} nT\n'
                    f'R²: {r2:.3f}\n'
                    f'Correlation: {correlation:.3f}\n'
                    f'Slope: {slope:.3f}\n'
                    f'N points: {len(true_y):,}')

    ax.text(0.98, 0.02, metrics_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()

    if save_path:
        save_file = f"{save_path}_scatter.png"
        plt.savefig(save_file, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_file}")

    return fig


def plot_dst_both(time_stamps, true_y, pred_y, title="Dst Predictions",
                  ts_figsize=(16, 8), scatter_figsize=(10, 10), scatter_storm_threshold=True,
                  series_storm_threshold=True, hexbin_plot=False, save_path=None, dpi=300):
    ts_fig = plot_dst_timeseries(time_stamps, true_y, pred_y, title,
                                 ts_figsize, series_storm_threshold, save_path, dpi)
    scatter_fig = plot_dst_scatter(time_stamps, true_y, pred_y, title,
                                   scatter_figsize, scatter_storm_threshold, save_path, dpi,
                                   hexbin_plot=hexbin_plot)
    return ts_fig, scatter_fig


# ============ EXAMPLE USAGE ============

def demo_dst_visualization():
    """
    Demonstrate the DST visualization functions with synthetic data.
    """
    # Generate synthetic DST-like data
    np.random.seed(42)
    n_points = 5000
    
    # Create realistic DST time series with storms
    time = np.linspace(0, 365, n_points)  # One year of hourly data
    
    # Base DST level with daily variation
    base_dst = -10 + 5 * np.sin(2 * np.pi * time / 24)  # Diurnal variation
    
    # Add storm events (sudden drops)
    storms = np.random.exponential(30, size=20)  # Random storm times
    for storm_time in storms:
        if storm_time < 365:
            storm_idx = int(storm_time * n_points / 365)
            storm_duration = np.random.randint(6, 24)  # 6-24 hour storms
            storm_magnitude = np.random.uniform(-50, -200)  # Storm intensity
            
            # Create storm profile (rapid onset, gradual recovery)
            for i in range(min(storm_duration, n_points - storm_idx)):
                if storm_idx + i < n_points:
                    # Exponential recovery
                    base_dst[storm_idx + i] += storm_magnitude * np.exp(-i/8)
    
    # Add noise
    true_dst = base_dst + np.random.normal(0, 5, n_points)
    
    # Create predictions with some error
    pred_dst = true_dst + np.random.normal(0, 8, n_points)  # Add prediction error
    pred_dst = 0.9 * pred_dst + 0.1 * np.mean(true_dst)  # Add slight bias
    
    # Plot both separately
    print("Creating time series plot...")
    ts_fig = plot_dst_timeseries(
        true_y=true_dst,
        pred_y=pred_dst, 
        start_date='2023-01-01',
        end_date='2023-12-31',
        title='Test Set',
        save_path='dst_demo'
    )
    
    print("Creating scatter plot...")
    scatter_fig = plot_dst_scatter(
        true_y=true_dst,
        pred_y=pred_dst, 
        start_date='2023-01-01',
        end_date='2023-12-31',
        title='Test Set',
        save_path='dst_demo'
    )
    
    plt.show()
    return ts_fig, scatter_fig

# Uncomment to run demo:
# demo_dst_visualization()


import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def create_transformation_comparison(left_values, left_name, right_values, right_name, 
                                   title="Distribution Transformation Comparison",
                                   figsize=(12, 5), bins=50, alpha=0.7,
                                   left_color='#2E8B57', right_color='#4682B4',
                                   save_path=None, dpi=300, show_stats=True):
    """
    Create publication-quality side-by-side histograms for transformation comparison.
    
    Parameters:
    -----------
    left_values : array-like
        Data for left histogram (usually original data)
    left_name : str
        Label for left histogram
    right_values : array-like
        Data for right histogram (usually transformed data)
    right_name : str
        Label for right histogram
    title : str
        Main title for the figure
    figsize : tuple
        Figure size (width, height)
    bins : int or str
        Number of bins or binning strategy
    alpha : float
        Transparency level for histograms
    left_color : str
        Color for left histogram
    right_color : str
        Color for right histogram
    save_path : str, optional
        Path to save the figure (if None, figure is shown)
    dpi : int
        Resolution for saved figure
    show_stats : bool
        Whether to show distribution statistics on plots
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes objects
    """
    
    # Set up the figure with publication-quality settings
    plt.style.use('seaborn-v0_8-whitegrid')  # Clean, professional look
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=False)
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    # Remove NaN values for plotting and statistics
    left_clean = np.array(left_values)[~np.isnan(left_values)]
    right_clean = np.array(right_values)[~np.isnan(right_values)]
    
    # Calculate statistics if requested
    if show_stats:
        left_stats = {
            'mean': np.mean(left_clean),
            'std': np.std(left_clean),
            'skew': stats.skew(left_clean),
            'kurt': stats.kurtosis(left_clean)
        }
        
        right_stats = {
            'mean': np.mean(right_clean),
            'std': np.std(right_clean),
            'skew': stats.skew(right_clean),
            'kurt': stats.kurtosis(right_clean)
        }
    
    # Left histogram (original data)
    n1, bins1, patches1 = ax1.hist(left_clean, bins=bins, alpha=alpha, 
                                   color=left_color, edgecolor='black', linewidth=0.5)
    ax1.set_title(f'{left_name}', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('Value', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Add statistics text box for left plot
    if show_stats:
        stats_text1 = f'Mean: {left_stats["mean"]:.2e}\nStd: {left_stats["std"]:.2e}\nSkew: {left_stats["skew"]:.2f}\nKurt: {left_stats["kurt"]:.2f}'
        ax1.text(0.75, 0.98, stats_text1, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Right histogram (transformed data)
    n2, bins2, patches2 = ax2.hist(right_clean, bins=bins, alpha=alpha,
                                   color=right_color, edgecolor='black', linewidth=0.5)
    ax2.set_title(f'{right_name}', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlabel('Value', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Add statistics text box for right plot
    if show_stats:
        stats_text2 = f'Mean: {right_stats["mean"]:.2e}\nStd: {right_stats["std"]:.2e}\nSkew: {right_stats["skew"]:.2f}\nKurt: {right_stats["kurt"]:.2f}'
        ax2.text(0.75, 0.98, stats_text2, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Improve layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Make room for main title
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()
    
    return fig, (ax1, ax2)


# Example usage function for space weather data
def plot_space_weather_transformation(original_data, transformed_data, 
                                     parameter_name="SW Plasma Temperature",
                                     original_units="K", transformed_units="normalized",
                                     save_path=None):
    """
    Specific function for space weather parameter transformations.
    """
    
    left_name = f"Original {parameter_name}\n({original_units})"
    right_name = f"Transformed {parameter_name}\n({transformed_units})"
    
    title = f"{parameter_name} Distribution Transformation"
    
    return create_transformation_comparison(
        left_values=original_data,
        left_name=left_name,
        right_values=transformed_data, 
        right_name=right_name,
        title=title,
        save_path=save_path
    )


# # Example with synthetic space weather-like data
# if __name__ == "__main__":
#     # Generate example data that mimics space weather characteristics
#     np.random.seed(42)
    
#     # Original: log-normal distribution (like SW temperature)
#     original = np.random.lognormal(mean=12, sigma=1.5, size=10000)  # 10³ to 10⁶ range
    
#     # Transformed: apply signed log transformation
#     transformed = np.sign(original) * np.log10(np.abs(original) + 1)
    
#     # Create the plot
#     fig, axes = plot_space_weather_transformation(
#         original_data=original,
#         transformed_data=transformed,
#         parameter_name="SW Plasma Temperature",
#         original_units="K",
#         transformed_units="log-normalized"
#     )


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

def plot_takens_trajectories(upper_row_data, bottom_row_data, 
                           parameter_name="SW Plasma Speed", 
                           units="km/s",
                           figsize=(15, 8),
                           dpi=300,
                           trajectory_count=6,
                           save_path=None):
    """
    Plot Takens embedding trajectories for quiet and storm periods.
    
    Parameters:
    -----------
    upper_row_data : list of arrays
        List of 6 trajectory arrays for quiet periods. Each array shape: (n_points, n_pairs, 2)
    bottom_row_data : list of arrays  
        List of 6 trajectory arrays for storm periods. Each array shape: (n_points, n_pairs, 2)
    parameter_name : str
        Name of the parameter being plotted
    units : str
        Units of the parameter
    figsize : tuple
        Figure size (width, height)
    dpi : int
        Figure resolution
    save_path : str, optional
        Path to save the figure
    """
    
    # Set up the figure with professional styling
    plt.style.use('default')  # Reset any custom styles
    fig, axes = plt.subplots(2, trajectory_count, figsize=figsize, dpi=dpi)
    
    # Color scheme
    quiet_color = '#2E86AB'    # Professional blue
    storm_color = '#A23B72'    # Professional magenta
    background_color = '#F8F9FA'
    
    # Set background color
    fig.patch.set_facecolor('white')
    
    # Plot quiet periods (top row)
    for i, trajectory_array in enumerate(upper_row_data):
        ax = axes[0, i]
        
        # Extract first trajectory pair for plotting (assuming we want the first 2D projection)
        if len(trajectory_array) > 0 and len(trajectory_array[0]) > 0:
            # Get the first 2D pair trajectory
            trajectory = trajectory_array[0]  # Shape: (n_time_points, 2)
            
            if len(trajectory) > 1:
                ax.plot(trajectory[:, 0], trajectory[:, 1], 
                       color=quiet_color, linewidth=1.5, alpha=0.8)
                
                # Add start and end markers
                ax.scatter(trajectory[0, 0], trajectory[0, 1], 
                          color=quiet_color, s=30, marker='o', zorder=5, alpha=0.9)
                ax.scatter(trajectory[-1, 0], trajectory[-1, 1], 
                          color=quiet_color, s=30, marker='s', zorder=5, alpha=0.9)
        
        # Styling
        ax.set_facecolor(background_color)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.tick_params(labelsize=8)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)
        
        # Labels only for leftmost plots
        if i == 0:
            ax.set_ylabel(f'{parameter_name}(t-τ)\n[{units}]', fontsize=10, fontweight='medium')
        
        # Title only for top row
        ax.set_title(f'Quiet Period {i+1}', fontsize=10, fontweight='medium', pad=8)
    
    # Plot storm periods (bottom row)
    for i, trajectory_array in enumerate(bottom_row_data):
        ax = axes[1, i]
        
        # Extract first trajectory pair for plotting
        if len(trajectory_array) > 0 and len(trajectory_array[0]) > 0:
            # Get the first 2D pair trajectory
            trajectory = trajectory_array[0]  # Shape: (n_time_points, 2)
            
            if len(trajectory) > 1:
                ax.plot(trajectory[:, 0], trajectory[:, 1], 
                       color=storm_color, linewidth=1.5, alpha=0.8)
                
                # Add start and end markers
                ax.scatter(trajectory[0, 0], trajectory[0, 1], 
                          color=storm_color, s=30, marker='o', zorder=5, alpha=0.9)
                ax.scatter(trajectory[-1, 0], trajectory[-1, 1], 
                          color=storm_color, s=30, marker='s', zorder=5, alpha=0.9)
        
        # Styling
        ax.set_facecolor(background_color)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.tick_params(labelsize=8)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)
        
        # Labels
        if i == 0:
            ax.set_ylabel(f'{parameter_name}(t-τ)\n[{units}]', fontsize=10, fontweight='medium')
        
        ax.set_xlabel(f'{parameter_name}(t)\n[{units}]', fontsize=10, fontweight='medium')
        ax.set_title(f'Storm Period {i+1}', fontsize=10, fontweight='medium', pad=8)
    
    # Add overall title and legend
    fig.suptitle(f'Phase Space Trajectories: {parameter_name}', 
                 fontsize=9, fontweight='bold', y=0.95)
    
    # Create custom legend
    quiet_patch = mpatches.Patch(color=quiet_color, label='Quiet Periods')
    storm_patch = mpatches.Patch(color=storm_color, label='Storm Periods')
    start_marker = plt.Line2D([0], [0], marker='o', color='gray', linewidth=0, 
                             markersize=6, label='Start Point')
    end_marker = plt.Line2D([0], [0], marker='s', color='gray', linewidth=0, 
                           markersize=6, label='End Point')
    
    fig.legend(handles=[quiet_patch, storm_patch, start_marker, end_marker],
               loc='center', bbox_to_anchor=(0.5, 0.02), ncol=4, fontsize=10,
               frameon=False)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, bottom=0.12, hspace=0.35, wspace=0.25)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    return fig, axes

# Example usage:
# plot_takens_trajectories(
#     upper_row_data=quiet_trajectories,  # List of 6 trajectory arrays
#     bottom_row_data=storm_trajectories, # List of 6 trajectory arrays
#     parameter_name="SW Plasma Speed",
#     units="km/s",
#     save_path="takens_trajectories.png"
# )


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import DateFormatter
import pandas as pd

def plot_convolution_comparison(original_series, convolved_series, storm_mask,
                              timestamps=None, measure_name="BZ", units="nT", 
                              kernel_name="Storm Onset", figsize=(12, 8), dpi=300, 
                              save_path=None):
    """
    Create a professional two-panel visualization showing before and after convolution with storm periods.
    
    Parameters:
    -----------
    original_series : array-like
        Original time series data
    convolved_series : array-like  
        Convolved time series data
    storm_mask : array-like
        Boolean array indicating storm periods (True = storm, False = quiet)
    timestamps : array-like, optional
        Time indices for x-axis. If None, uses integer indices
    measure_name : str
        Name of the measured parameter
    units : str
        Units of the original measurement
    kernel_name : str
        Name of the convolution kernel used
    figsize : tuple
        Figure size (width, height)
    dpi : int
        Figure resolution
    save_path : str, optional
        Path to save the figure
    """
    
    # Set up professional styling
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, dpi=dpi, sharex=True)
    
    # Color scheme
    original_color = '#2E86AB'      # Professional blue
    convolved_color = '#A23B72'     # Professional magenta
    storm_color = '#FFB6C1'         # Light red for storm background
    grid_color = '#E5E5E5'
    text_color = '#333333'
    response_color = '#FF6B6B'      # Bright red for significant responses
    
    # Set up time axis
    if timestamps is not None:
        x_axis = timestamps
        xlabel = 'Time'
    else:
        x_axis = np.arange(len(original_series))
        xlabel = 'Time Index (hours)'
    
    # Ensure arrays are the same length
    min_length = min(len(original_series), len(convolved_series), len(storm_mask), len(x_axis))
    original_series = np.array(original_series[:min_length])
    convolved_series = np.array(convolved_series[:min_length])
    storm_mask = np.array(storm_mask[:min_length], dtype=bool)
    x_axis = x_axis[:min_length]
    
    # Plot storm periods as background shading for both panels
    if np.any(storm_mask):
        # Find continuous storm periods for cleaner visualization
        storm_starts = []
        storm_ends = []
        in_storm = False
        
        for i, is_storm in enumerate(storm_mask):
            if is_storm and not in_storm:
                storm_starts.append(i)
                in_storm = True
            elif not is_storm and in_storm:
                storm_ends.append(i)
                in_storm = False
        
        # Handle case where data ends during a storm
        if in_storm:
            storm_ends.append(len(storm_mask))
        
        # Add storm period shading to both axes
        for start, end in zip(storm_starts, storm_ends):
            ax1.axvspan(x_axis[start], x_axis[end-1], alpha=0.2, color=storm_color, zorder=0)
            ax2.axvspan(x_axis[start], x_axis[end-1], alpha=0.2, color=storm_color, zorder=0)
    
    # Plot original time series (top panel)
    ax1.plot(x_axis, original_series, color=original_color, linewidth=1.2, alpha=0.8)
    ax1.fill_between(x_axis, original_series, alpha=0.2, color=original_color)
    
    # Styling for top panel
    ax1.set_ylabel(f'{measure_name}\n[{units}]', fontsize=11, fontweight='medium', color=text_color)
    ax1.set_title(f'Scaled time series: {measure_name}', fontsize=12, fontweight='bold', 
                  color=text_color, pad=15)
    ax1.grid(True, alpha=0.3, color=grid_color, linewidth=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color('#CCCCCC')
    ax1.spines['bottom'].set_color('#CCCCCC')
    ax1.tick_params(colors=text_color, labelsize=9)
    
    # Plot convolved time series (bottom panel)
    ax2.plot(x_axis, convolved_series, color=convolved_color, linewidth=1.2, alpha=0.8)
    ax2.fill_between(x_axis, convolved_series, alpha=0.2, color=convolved_color)
    
    # Highlight significant responses (above 2 standard deviations)
    conv_threshold = np.mean(convolved_series) + 2 * np.std(convolved_series)
    significant_mask = convolved_series > conv_threshold
    if np.any(significant_mask):
        ax2.scatter(x_axis[significant_mask], convolved_series[significant_mask], 
                   color=response_color, s=25, zorder=5, alpha=0.9, 
                   label=f'High Response (>{conv_threshold:.2f})')
        ax2.legend(loc='upper right', fontsize=9)
    
    # Styling for bottom panel
    ax2.set_ylabel(f'Convolved Response\n[dimensionless]', fontsize=11, fontweight='medium', color=text_color)
    ax2.set_xlabel(xlabel, fontsize=11, fontweight='medium', color=text_color)
    ax2.set_title(f'After {kernel_name} Kernel Convolution', fontsize=12, fontweight='bold', 
                  color=text_color, pad=15)
    ax2.grid(True, alpha=0.3, color=grid_color, linewidth=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color('#CCCCCC')
    ax2.spines['bottom'].set_color('#CCCCCC')
    ax2.tick_params(colors=text_color, labelsize=9)
    
    # Overall title
    # fig.suptitle(f'{measure_name}', 
    #              fontsize=14, fontweight='bold', y=0.95, color=text_color)
    
    # Add legend for storm periods (only if storms exist)
    if np.any(storm_mask):
        import matplotlib.patches as mpatches
        storm_patch = mpatches.Patch(color=storm_color, alpha=0.6, label='DST Index values <= -50')
        
        # Add to existing legend or create new one
        if np.any(significant_mask):
            # Get existing legend elements and add storm patch
            handles, labels = ax2.get_legend_handles_labels()
            handles.append(storm_patch)
            labels.append('DST Index values <= -50')
            ax2.legend(handles, labels, loc='upper right', fontsize=9)
        else:
            ax2.legend(handles=[storm_patch], loc='upper right', fontsize=9)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, hspace=0.3)
    
    # Add subtle background
    fig.patch.set_facecolor('#FAFAFA')
    
    # Format x-axis if using datetime
    if timestamps is not None and hasattr(timestamps, 'dtype') and 'datetime' in str(timestamps.dtype):
        ax2.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    return fig, (ax1, ax2)

# Example usage:
# plot_convolution_comparison(
#     original_series=df['BZ, nT (GSM)'].values,
#     convolved_series=df['conv_storm_onset_short_BZ_nT_GSM'].values,
#     storm_mask=df['dst_target'].values,  # assuming True = storm
#     timestamps=df.index,
#     measure_name="BZ (GSM)",
#     units="nT",
#     kernel_name="Storm Onset (Short)",
#     save_path="convolution_example.png"
# )