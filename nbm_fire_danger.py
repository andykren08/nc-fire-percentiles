import os
import requests
import numpy as np
import xarray as xr
import cfgrib
import glob
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --- 1. CONFIGURATION & THRESHOLDS ---
# Red Flag Warning / High Danger Thresholds
CRITICAL_RH = 25.0       # Percent
CRITICAL_WIND = 20.0     # MPH
CRITICAL_GUST = 30.0     # MPH

# Define NC Bounding Box (Widened for better visual buffer)
lat_min, lat_max = 32.0, 38.0
lon_min, lon_max = -86.5, -73.0

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

def generate_prob_plot(plot_data, lats, lons, fhr, scenario, title_text, init_time, is_uncertainty=False):
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    # Add Map Features
    ax.add_feature(cfeature.STATES, linewidth=1.5, edgecolor='black')
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0)
    
    try:
        from metpy.plots import USCOUNTIES
        ax.add_feature(USCOUNTIES.with_scale('5m'), edgecolor='gray', linewidth=0.5)
    except:
        pass

    # Setup Colors & Legend Ticks
    if is_uncertainty:
        cmap = plt.cm.plasma  
        levels = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
        tick_locs = [1, 2, 3, 4, 5]
        tick_labels = ['1 (Low)', '2', '3', '4', '5 (High)']
    else:
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(['#A1D99B', '#FEE08B', '#D73027']) # Green, Yellow, Red
        levels = [-0.5, 0.5, 1.5, 2.5]
        tick_locs = [0, 1, 2]
        tick_labels = ['Low', 'Elevated', 'Critical']

    # Plot the Data
    mesh = ax.pcolormesh(lons, lats, plot_data, cmap=cmap, vmin=levels[0], vmax=levels[-1], transform=ccrs.PlateCarree())
    
    # Formatting the Colorbar
    cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.05, aspect=50, ticks=tick_locs)
    cbar.set_ticklabels(tick_labels)
    
    # Calculate and Format the Valid Time
    valid_time = init_time + timedelta(hours=fhr)
    valid_time_str = valid_time.strftime("%a %m/%d %H:00Z")
    
    plt.title(f"NBM {title_text}\nValid: {valid_time_str} (F{fhr:02d})", fontsize=14, fontweight='bold')
    
    # Save Image
    filename = f"public/images/nbm_{scenario}_fire_danger_f{fhr:02d}.png"
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved {filename}")

# --- 5. MAIN NBM PROCESSING LOOP ---
def process_nbm():
    print("--- Processing NBM Percentiles ---")
    
    # NBM QMD (probabilistic) runs happen at 00Z, 06Z, 12Z, and 18Z.
    # It takes about 2.5 hours for them to fully upload to NOMADS. 
    # Subtract 4 hours from current time to ensure we look for a FINISHED run.
    now = datetime.utcnow()
    safe_time = now - timedelta(hours=4)
    
    # Round down to the nearest 00, 06, 12, or 18 cycle
    cycle_hour = (safe_time.hour // 6) * 6
    
    date_str = safe_time.strftime("%Y%m%d")
    hour_str = f"{cycle_hour:02d}"
    
    init_time = datetime(safe_time.year, safe_time.month, safe_time.day, cycle_hour, 0)
    
    print(f"Current UTC: {now.strftime('%H:%Mz')} | Targeting NBM Cycle: {date_str} at {hour_str}Z")
    
    base_url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/blend/prod/blend.{date_str}/{hour_str}/qmd"
    
    for fhr in range(1, 49):
        file_name = f"blend.t{hour_str}z.qmd.f{fhr:03d}.co.grib2"
        file_url = f"{base_url}/{file_name}"
        
        # Initialize datasets variable so the finally block doesn't crash if download fails
        datasets = []
        
        try:
            print(f"Downloading: {file_url}")
            
            # NOAA requires a User-Agent to accept the connection
            headers = {'User-Agent': 'NC-Forest-Service-Probabilistic-App'}
            response = requests.get(file_url, headers=headers, timeout=30)
            
            if response.status_code != 200:
                print(f" -> File not available yet (Status {response.status_code})")
                continue
                
            with open(file_name, 'wb') as f:
                f.write(response.content)
                
            # Open ALL hypercubes inside the GRIB file
            datasets = cfgrib.open_datasets(file_name)
            
            rh_10 = rh_90 = wind_10 = wind_90 = None
            lats = lons = None

            # Dynamically hunt for the variables
            for d in datasets:
                if lats is None:
                    if 'latitude' in d.coords:
                        lats = d.latitude.values
                        lons = d.longitude.values
                    elif 'lat' in d.coords:
                        lats = d.lat.values
                        lons = d.lon.values

                for var in d.data_vars:
                    if 'percentile' in d.coords:
                        if var in ['r2', 'rh', '2r']:
                            try:
                                rh_10 = d[var].sel(percentile=10).values
                                rh_90 = d[var].sel(percentile=90).values
                            except: pass
                        if var in ['si10', 'wspd', '10si', 'wind', 'gust']:
                            try:
                                wind_10 = d[var].sel(percentile=10).values
                                wind_90 = d[var].sel(percentile=90).values
                            except: pass
                    else:
                        if var in ['r2_10', 'rh_10', '2r_10']: rh_10 = d[var].values
                        if var in ['r2_90', 'rh_90', '2r_90']: rh_90 = d[var].values
                        if var in ['si10_10', 'wspd_10', '10si_10']: wind_10 = d[var].values
                        if var in ['si10_90', 'wspd_90', '10si_90']: wind_90 = d[var].values

            if rh_10 is None or rh_90 is None:
                print(f" -> Could not find RH percentiles for Hour {fhr}")
                continue
                
            if wind_10 is None or wind_90 is None:
                print(f" -> Could not find Wind percentiles for Hour {fhr}")
                continue

            wind_10 = ms_to_mph(wind_10)
            wind_90 = ms_to_mph(wind_90)

            worst_case = calculate_fire_danger(rh_10, wind_90, wind_90)
            generate_prob_plot(worst_case, lats, lons, fhr, "worst", "Worst-Case Scenario (Low RH / High Wind)", init_time)
            
            best_case = calculate_fire_danger(rh_90, wind_10, wind_10)
            generate_prob_plot(best_case, lats, lons, fhr, "best", "Best-Case Scenario (High RH / Low Wind)", init_time)
            
            uncertainty = calculate_uncertainty_index(rh_10, rh_90, wind_10, wind_90)
            generate_prob_plot(uncertainty, lats, lons, fhr, "spread", "Forecast Uncertainty Index", init_time, is_uncertainty=True)
            
        except Exception as e:
            print(f"Error processing NBM Hour {fhr}: {e}")
            
        finally:
            # THIS IS THE MAGIC FIX: Runs no matter what!
            # 1. Force close any datasets that successfully opened
            for d in datasets:
                try: d.close()
                except: pass
                
            # 2. Nuke the GRIB file and any hidden .idx files
            for junk_file in glob.glob(f"{file_name}*"):
                try: os.remove(junk_file)
                except: pass

if __name__ == "__main__":
    process_nbm()
