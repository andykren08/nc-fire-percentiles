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
CRITICAL_RH = 25.0       # Percent
CRITICAL_WIND = 20.0     # MPH
CRITICAL_GUST = 30.0     # MPH

lat_min, lat_max = 33.0, 37.5
lon_min, lon_max = -85.5, -74.5

os.makedirs('public/images', exist_ok=True)

# --- 2. FIRE DANGER MATH ---
def calculate_fire_danger(rh, wind, gust):
    danger_grid = np.zeros_like(rh, dtype=int)
    elevated_mask = (rh <= CRITICAL_RH + 5) & ((wind >= CRITICAL_WIND - 5) | (gust >= CRITICAL_GUST - 5))
    danger_grid[elevated_mask] = 1
    critical_mask = (rh <= CRITICAL_RH) & ((wind >= CRITICAL_WIND) | (gust >= CRITICAL_GUST))
    danger_grid[critical_mask] = 2
    return danger_grid

def calculate_uncertainty_index(rh_10, rh_90, wind_10, wind_90):
    rh_spread = np.abs(rh_90 - rh_10)
    wind_spread = np.abs(wind_90 - wind_10)
    
    rh_score = np.ones_like(rh_spread, dtype=int)
    rh_score[rh_spread >= 10] = 2
    rh_score[rh_spread >= 20] = 3
    rh_score[rh_spread >= 30] = 4
    rh_score[rh_spread >= 40] = 5
    
    wind_score = np.ones_like(wind_spread, dtype=int)
    wind_score[wind_spread >= 5] = 2
    wind_score[wind_spread >= 10] = 3
    wind_score[wind_spread >= 15] = 4
    wind_score[wind_spread >= 20] = 5
    
    return np.ceil((rh_score + wind_score) / 2.0).astype(int)

def ms_to_mph(ms):
    return ms * 2.23694

# --- 3. MAPPING ---
def generate_prob_plot(plot_data, lats, lons, day, scenario, title_text, init_time, is_uncertainty=False):
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.STATES, linewidth=1.5, edgecolor='black')
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0)
    
    try:
        from metpy.plots import USCOUNTIES
        ax.add_feature(USCOUNTIES.with_scale('5m'), edgecolor='gray', linewidth=0.5)
    except: pass

    if is_uncertainty:
        cmap = plt.cm.plasma  
        levels = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
        tick_locs = [1, 2, 3, 4, 5]
        tick_labels = ['1 (Low)', '2', '3', '4', '5 (High)']
    else:
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(['#A1D99B', '#FEE08B', '#D73027']) 
        levels = [-0.5, 0.5, 1.5, 2.5]
        tick_locs = [0, 1, 2]
        tick_labels = ['Low', 'Elevated', 'Critical']

    mesh = ax.pcolormesh(lons, lats, plot_data, cmap=cmap, vmin=levels[0], vmax=levels[-1], transform=ccrs.PlateCarree())
    
    cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.05, aspect=50, ticks=tick_locs)
    cbar.set_ticklabels(tick_labels)
    
    valid_time = init_time + timedelta(hours=(day * 24))
    plt.title(f"NBM {title_text}\nValid 24h Ending: {valid_time.strftime('%a %m/%d %H:00Z')} (Day {day})", fontsize=14, fontweight='bold')
    
    filename = f"public/images/nbm_{scenario}_fire_danger_day{day}.png"
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close()

# --- 4. MAIN PROCESSING LOOP ---
def process_nbm():
    print("--- Hunting for NBM Major Cycles (7-Day Strategic Outlook) ---")
    now = datetime.utcnow()
    valid_cycle_found = False
    
    for hours_back in range(0, 48):
        check_time = now - timedelta(hours=hours_back)
        cycle_hour = check_time.hour
        
        # Target the Major Cycles that contain 24-hour daily summaries
        if cycle_hour not in [1, 7, 13, 19]:
            continue
            
        date_str = check_time.strftime("%Y%m%d")
        hour_str = f"{cycle_hour:02d}"
        
        # Test f024 (Day 1) to ensure the cycle has uploaded
        test_url = f"https://noaa-nbm-grib2-pds.s3.amazonaws.com/blend.{date_str}/{hour_str}/qmd/blend.t{hour_str}z.qmd.f024.co.grib2"
        
        try:
            if requests.head(test_url, timeout=10).status_code == 200:
                print(f"Success! Locked onto fresh NBM Major Cycle: {date_str} at {hour_str}Z")
                valid_cycle_found = True
                break
        except: pass 

    if not valid_cycle_found:
        print("CRITICAL: Could not find any uploaded NBM Major cycles.")
        return

    init_time = datetime(check_time.year, check_time.month, check_time.day, cycle_hour, 0)
    base_url = f"https://noaa-nbm-grib2-pds.s3.amazonaws.com/blend.{date_str}/{hour_str}/qmd"
    
    # Loop through Day 1 to Day 7 (Forecast Hours 24, 48, 72, 96, 120, 144, 168)
    for day in range(1, 8):
        fhr = day * 24
        file_name = f"blend.t{hour_str}z.qmd.f{fhr:03d}.co.grib2"
        file_url = f"{base_url}/{file_name}"
        datasets = []
        
        try:
            print(f"Downloading Day {day} (F{fhr:03d})...")
            resp = requests.get(file_url, timeout=30)
            if resp.status_code != 200: 
                print(f" -> Day {day} not available.")
                continue
                
            with open(file_name, 'wb') as f:
                f.write(resp.content)
                
            datasets = cfgrib.open_datasets(file_name)
            
            rh_10 = rh_90 = wind_10 = wind_90 = lats = lons = None

            for d in datasets:
                if lats is None and 'latitude' in d.coords:
                    lats = d.latitude.values
                    lons = d.longitude.values
                elif lats is None and 'lat' in d.coords:
                    lats = d.lat.values
                    lons = d.lon.values

                for var in d.data_vars:
                    # Hunt for Minimum RH (Worst case is 10th percentile, Best case is 90th)
                    if 'percentile' in d.coords:
                        if var in ['minrh', 'minrh2m', 'rh', '2r']:
                            try:
                                rh_10 = d[var].sel(percentile=10).values
                                rh_90 = d[var].sel(percentile=90).values
                            except: pass
                        # Hunt for Maximum Wind/Gust (Worst case is 90th percentile, Best case is 10th)
                        if var in ['maxwind', 'maxwspd', 'maxg', 'gust', 'si10', 'wind']:
                            try:
                                wind_10 = d[var].sel(percentile=10).values
                                wind_90 = d[var].sel(percentile=90).values
                            except: pass
                    else:
                        if var in ['minrh_10', 'minrh2m_10', 'rh_10']: rh_10 = d[var].values
                        if var in ['minrh_90', 'minrh2m_90', 'rh_90']: rh_90 = d[var].values
                        if var in ['maxwind_10', 'maxwspd_10', 'wind_10']: wind_10 = d[var].values
                        if var in ['maxwind_90', 'maxwspd_90', 'wind_90']: wind_90 = d[var].values

            if rh_10 is None or wind_90 is None:
                print(f" -> Could not find Min RH or Max Wind percentiles for Day {day}")
                continue

            wind_10 = ms_to_mph(wind_10)
            wind_90 = ms_to_mph(wind_90)

            # Worst-Case: Driest Min RH (10th) and Windiest Max Wind (90th)
            worst_case = calculate_fire_danger(rh_10, wind_90, wind_90)
            generate_prob_plot(worst_case, lats, lons, day, "worst", "Worst-Case Daily Peak Danger", init_time)
            
            # Best-Case: Wettest Min RH (90th) and Calmest Max Wind (10th)
            best_case = calculate_fire_danger(rh_90, wind_10, wind_10)
            generate_prob_plot(best_case, lats, lons, day, "best", "Best-Case Daily Peak Danger", init_time)
            
            uncertainty = calculate_uncertainty_index(rh_10, rh_90, wind_10, wind_90)
            generate_prob_plot(uncertainty, lats, lons, day, "spread", "7-Day Uncertainty Spread", init_time, is_uncertainty=True)
            
        except Exception as e:
            print(f"Error processing Day {day}: {e}")
            
        finally:
            for d in datasets:
                try: d.close()
                except: pass
            for junk in glob.glob(f"{file_name}*"):
                try: os.remove(junk)
                except: pass

if __name__ == "__main__":
    process_nbm()
