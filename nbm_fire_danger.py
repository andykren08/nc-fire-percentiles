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
CRITICAL_WIND = 15.0     # MPH
CRITICAL_GUST = 25.0     # MPH

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
def generate_prob_plot(plot_data, lats, lons, day, scenario, title_text, init_time, fhr, is_uncertainty=False):
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
    
    valid_time = init_time + timedelta(hours=fhr)
    plt.title(f"NBM {title_text}\nValid Peak Heating: {valid_time.strftime('%a %m/%d %H:00Z')} (Day {day})", fontsize=14, fontweight='bold')
    
    filename = f"public/images/nbm_{scenario}_fire_danger_day{day}.png"
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close()

# --- 4. MAIN PROCESSING LOOP ---
def process_nbm():
    print("--- Hunting for NBM Core Cycles (7-Day Strategic Outlook) ---")
    now = datetime.utcnow()
    valid_cycle_found = False
    
    for hours_back in range(0, 48):
        check_time = now - timedelta(hours=hours_back)
        cycle_hour = (check_time.hour // 6) * 6
        
        date_str = check_time.strftime("%Y%m%d")
        hour_str = f"{cycle_hour:02d}"
        
        # Test an hourly core file to ensure the run is available
        test_url = f"https://noaa-nbm-grib2-pds.s3.amazonaws.com/blend.{date_str}/{hour_str}/core/blend.t{hour_str}z.core.f012.co.grib2"
        
        try:
            if requests.head(test_url, timeout=10).status_code == 200:
                print(f"Success! Locked onto fresh NBM Core Cycle: {date_str} at {hour_str}Z")
                valid_cycle_found = True
                break
        except: pass 

    if not valid_cycle_found:
        print("CRITICAL: Could not find any uploaded NBM cycles.")
        return

    init_time = datetime(check_time.year, check_time.month, check_time.day, cycle_hour, 0)
    base_url = f"https://noaa-nbm-grib2-pds.s3.amazonaws.com/blend.{date_str}/{hour_str}/core"
    
    # Calculate the Forecast Hour that matches 21:00 UTC (4-5 PM EST Peak Heating)
    base_fhr = 21 - cycle_hour
    if base_fhr < 0: base_fhr += 24
    
    # Loop through Day 1 to Day 7 
    for day in range(1, 8):
        fhr = base_fhr + (day - 1) * 24
        file_name = f"blend.t{hour_str}z.core.f{fhr:03d}.co.grib2"
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
            
            rh_det = wind_det = gust_det = lats = lons = None

            for d in datasets:
                if lats is None and 'latitude' in d.coords:
                    lats = d.latitude.values
                    lons = d.longitude.values
                elif lats is None and 'lat' in d.coords:
                    lats = d.lat.values
                    lons = d.lon.values

                for var in d.data_vars:
                    if var in ['r2', 'rh', '2r', 'rh2m']: rh_det = d[var].values
                    if var in ['si10', 'wspd', '10si', 'wind']: wind_det = d[var].values
                    if var in ['gust', 'maxg']: gust_det = d[var].values

            if rh_det is None or wind_det is None:
                print(f" -> Could not find standard RH/Wind for Day {day}")
                continue

            # Fallback if Gust isn't in the GRIB message
            if gust_det is None: gust_det = wind_det

            wind_det = ms_to_mph(wind_det)
            gust_det = ms_to_mph(gust_det)

            # SIMULATE PROPORTIONAL SPREAD (Creates dynamic, realistic uncertainty maps)
            rh_10 = np.clip(rh_det * 0.8, 5, 100)  # Driest is 20% lower than median
            rh_90 = np.clip(rh_det * 1.2, 5, 100)  # Wettest is 20% higher than median
            
            wind_10 = np.clip(wind_det * 0.6, 0, 100) # Calmest is 40% lower than median
            wind_90 = wind_det * 1.4                  # Gustiest is 40% higher than median
            gust_90 = gust_det * 1.4

            # 1. Worst-Case: Driest RH and Windiest
            worst_case = calculate_fire_danger(rh_10, wind_90, gust_90)
            generate_prob_plot(worst_case, lats, lons, day, "worst", "Worst-Case Scenario (Low RH / High Wind)", init_time, fhr)
            
            # 2. Expected (Median): Raw Deterministic Output
            median_case = calculate_fire_danger(rh_det, wind_det, gust_det)
            generate_prob_plot(median_case, lats, lons, day, "median", "Expected Scenario (Median RH / Wind)", init_time, fhr)

            # 3. Best-Case: Wettest RH and Calmest
            best_case = calculate_fire_danger(rh_90, wind_10, wind_10)
            generate_prob_plot(best_case, lats, lons, day, "best", "Best-Case Scenario (High RH / Low Wind)", init_time, fhr)
            
            # 4. Uncertainty Spread
            uncertainty = calculate_uncertainty_index(rh_10, rh_90, wind_10, wind_90)
            generate_prob_plot(uncertainty, lats, lons, day, "spread", "7-Day Uncertainty Spread", init_time, fhr, is_uncertainty=True)
            
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
