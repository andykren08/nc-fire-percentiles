import os
import requests
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --- 1. CONFIGURATION & THRESHOLDS ---
# Red Flag Warning / High Danger Thresholds
CRITICAL_RH = 25.0       # Percent
CRITICAL_WIND = 15.0     # MPH
CRITICAL_GUST = 25.0     # MPH

# Define NC Bounding Box
lat_min, lat_max = 33.5, 36.6
lon_min, lon_max = -84.5, -75.0

# Set up directories
os.makedirs('public/images', exist_ok=True)
os.makedirs('data', exist_ok=True)

# --- 2. THE FIRE DANGER MATH (Worst/Best Case) ---
def calculate_fire_danger(rh, wind, gust):
    """
    0 = Low, 1 = Elevated (approaching criteria), 2 = Critical (Red Flag)
    """
    danger_grid = np.zeros_like(rh, dtype=int)
    
    # Elevated Danger: Near critical thresholds
    elevated_mask = (rh <= CRITICAL_RH + 5) & ((wind >= CRITICAL_WIND - 5) | (gust >= CRITICAL_GUST - 5))
    danger_grid[elevated_mask] = 1
    
    # Critical Danger: Exceeds thresholds
    critical_mask = (rh <= CRITICAL_RH) & ((wind >= CRITICAL_WIND) | (gust >= CRITICAL_GUST))
    danger_grid[critical_mask] = 2
    
    return danger_grid

# --- 3. THE UNCERTAINTY INDEX MATH ---
def calculate_uncertainty_index(rh_10, rh_90, wind_10, wind_90):
    """
    Calculates a 1-5 confidence/uncertainty score.
    1 = High Confidence (Low Spread)
    5 = Low Confidence (High Spread / Boom or Bust)
    """
    # Calculate raw spreads
    rh_spread = np.abs(rh_90 - rh_10)
    wind_spread = np.abs(wind_90 - wind_10)
    
    # RH Score Matrix (1 to 5)
    rh_score = np.ones_like(rh_spread, dtype=int)
    rh_score[rh_spread >= 10] = 2
    rh_score[rh_spread >= 20] = 3
    rh_score[rh_spread >= 30] = 4
    rh_score[rh_spread >= 40] = 5
    
    # Wind Score Matrix (1 to 5)
    wind_score = np.ones_like(wind_spread, dtype=int)
    wind_score[wind_spread >= 5] = 2
    wind_score[wind_spread >= 10] = 3
    wind_score[wind_spread >= 15] = 4
    wind_score[wind_spread >= 20] = 5
    
    # Average them together (round up to be safe)
    combined_index = np.ceil((rh_score + wind_score) / 2.0).astype(int)
    
    return combined_index

# --- 4. UTILITIES & MAPPING SETUP ---
def ms_to_mph(ms):
    return ms * 2.23694

def generate_prob_plot(plot_data, lats, lons, fhr, scenario, title_text, is_uncertainty=False):
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    # Add Map Features (Matching your original script)
    ax.add_feature(cfeature.STATES, linewidth=1.5, edgecolor='black')
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0)
    
    # Optional: MetPy Counties (If you want to keep them, ensure MetPy is imported and shapefiles downloaded)
    try:
        from metpy.plots import USCOUNTIES
        ax.add_feature(USCOUNTIES.with_scale('5m'), edgecolor='gray', linewidth=0.5)
    except:
        pass

    # Setup Colors
    if is_uncertainty:
        # Uncertainty Index: 1 (Low) to 5 (High)
        cmap = plt.cm.plasma  # Uses a vibrant purple-to-yellow colormap
        levels = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    else:
        # Fire Danger: 0 (Low), 1 (Elevated), 2 (Critical)
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(['#A1D99B', '#FEE08B', '#D73027']) # Green, Yellow, Red
        levels = [-0.5, 0.5, 1.5, 2.5]

    # Plot the Data
    mesh = ax.pcolormesh(lons, lats, plot_data, cmap=cmap, vmin=levels[0], vmax=levels[-1], transform=ccrs.PlateCarree())
    
    # Formatting
    plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.05, aspect=50, ticks=[1, 2, 3, 4, 5] if is_uncertainty else [0, 1, 2])
    plt.title(f"NBM {title_text} - Forecast Hour {fhr:02d}", fontsize=14, fontweight='bold')
    
    # Save Image
    filename = f"public/images/nbm_{scenario}_fire_danger_f{fhr:02d}.png"
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved {filename}")

# --- 5. MAIN NBM PROCESSING LOOP ---
def process_nbm():
    print("--- Processing NBM Percentiles ---")
    
    # NBM URLs for NOMADS (Using a simplified core/qmd access pattern)
    date_str = datetime.utcnow().strftime("%Y%m%d")
    hour_str = "06" # We'll default to the 06Z run which has all percentiles
    
    base_url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/blend/prod/blend.{date_str}/{hour_str}/qmd"
    
    for fhr in range(1, 49):
        # Note: NBM probabilistic variables are usually packed into the 'qmd' or 'core' files depending on the NWS update cycle.
        # For this setup, we'll simulate the data extraction block using our dummy NOMADS pipeline.
        
        try:
            # 1. Download the file for this hour (Hypothetical grib2 file handling)
            # file_url = f"{base_url}/blend.t{hour_str}z.qmd.f{fhr:03d}.co.grib2"
            # ds = xr.open_dataset('downloaded_file.grib2', engine='cfgrib')
            
            # 2. Extract Data (Simulated arrays for testing the logic matrix)
            # In production, you will pull:
            # rh_10 = ds['rh10'].values
            # wind_90 = ds['si10_90'].values, etc.
            
            # --- SIMULATED DATA FOR NOW TO AVOID NOMADS TIMEOUTS DURING BUILD ---
            lats = np.linspace(lat_max, lat_min, 100)
            lons = np.linspace(lon_min, lon_max, 100)
            lon_grid, lat_grid = np.meshgrid(lons, lats)
            
            # Simulate 10th Percentile (Dry/Calm)
            rh_10 = np.random.uniform(15, 30, (100, 100))
            wind_10 = np.random.uniform(5, 10, (100, 100))
            
            # Simulate 90th Percentile (Wet/Windy)
            rh_90 = np.random.uniform(35, 60, (100, 100))
            wind_90 = np.random.uniform(15, 30, (100, 100))
            
            # 3. Calculate Scenarios
            # Worst Case: 10th RH, 90th Wind
            worst_case = calculate_fire_danger(rh_10, wind_90, wind_90)
            generate_prob_plot(worst_case, lats, lons, fhr, "worst", "Worst-Case Scenario (Low RH / High Wind)")
            
            # Best Case: 90th RH, 10th Wind
            best_case = calculate_fire_danger(rh_90, wind_10, wind_10)
            generate_prob_plot(best_case, lats, lons, fhr, "best", "Best-Case Scenario (High RH / Low Wind)")
            
            # Uncertainty Spread
            uncertainty = calculate_uncertainty_index(rh_10, rh_90, wind_10, wind_90)
            generate_prob_plot(uncertainty, lats, lons, fhr, "spread", "Forecast Uncertainty Index", is_uncertainty=True)
            
        except Exception as e:
            print(f"Error processing NBM Hour {fhr}: {e}")

if __name__ == "__main__":
    process_nbm()
