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
os.makedirs('data', exist_ok=True)

# --- 2. THERMODYNAMICS & FIRE DANGER MATH ---
def calculate_rh(t_k, td_k):
    """
    Calculates Relative Humidity from Temperature and Dewpoint in Kelvin
    using the August-Roche-Magnus approximation.
    """
    # Convert Kelvin to Celsius
    t_c = t_k - 273.15
    td_c = td_k - 273.15
    
    # Calculate Vapor Pressure (e) and Saturation Vapor Pressure (es)
    e = 6.112 * np.exp((17.67 * td_c) / (td_c + 243.5))
    es = 6.112 * np.exp((17.67 * t_c) / (t_c + 243.5))
    
    rh = 100.0 * (e / es)
    return np.clip(rh, 0, 100) # Ensure it stays realistically bounded between 0-100%

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
    
    combined_index = np.ceil((rh_score + wind_score) / 2.0).astype(int)
    return combined_index

# --- 3. UTILITIES & MAPPING ---
def ms_to_mph(ms):
    return ms * 2.23694

def generate_prob_plot(plot_data, lats, lons, fhr, scenario, title_text, init_time, is_uncertainty=False):
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
    plt.title(f"NBM {title_text}\nValid: {valid_time.strftime('%a %m/%d %H:00Z')} (F{fhr:02d})", fontsize=14, fontweight='bold')
    
    filename = f"public/images/nbm_{scenario}_fire_danger_f{fhr:02d}.png"
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close()

# --- 4. MAIN NBM PROCESSING LOOP ---
def process_nbm():
    print("--- Hunting for Latest NBM Probabilistic 'QMD' Cycle ---")
    now = datetime.utcnow()
    valid_cycle_found = False
    
    for hours_back in range(0, 48):
        check_time = now - timedelta(hours=hours_back)
        cycle_hour = check_time.hour
        
        # Check only the Major Probabilistic Cycles
        if cycle_hour not in [1, 7, 13, 19]:
            continue
            
        date_str = check_time.strftime("%Y%m%d")
        hour_str = f"{cycle_hour:02d}"
        
        # Pointing to the QMD bucket
        test_url = f"https://noaa-nbm-grib2-pds.s3.amazonaws.com/blend.{date_str}/{hour_str}/qmd/blend.t{hour_str}z.qmd.f001.co.grib2"
        
        try:
            if requests.head(test_url, timeout=10).status_code == 200:
                print(f"Success! Locked onto fresh NBM QMD Cycle: {date_str} at {hour_str}Z")
                valid_cycle_found = True
                break
        except: pass 

    if not valid_cycle_found:
        print("CRITICAL: Could not find any NBM QMD cycles.")
        return

    init_time = datetime(check_time.year, check_time.month, check_time.day, cycle_hour, 0)
    base_url = f"https://noaa-nbm-grib2-pds.s3.amazonaws.com/blend.{date_str}/{hour_str}/qmd"
    
    for fhr in range(1, 49):
        file_name = f"blend.t{hour_str}z.qmd.f{fhr:03d}.co.grib2"
        file_url = f"{base_url}/{file_name}"
        datasets = []
        
        try:
            print(f"Downloading: {file_url}")
            resp = requests.get(file_url, timeout=30)
            if resp.status_code != 200: continue
                
            with open(file_name, 'wb') as f:
                f.write(resp.content)
                
            datasets = cfgrib.open_datasets(file_name)
            
            t_10 = t_90 = td_10 = td_90 = wind_10 = wind_90 = None
            lats = lons = None

            # Dynamic hunter for T, Td, and Wind percentiles
            for d in datasets:
                if lats is None and 'latitude' in d.coords:
                    lats = d.latitude.values
                    lons = d.longitude.values
                elif lats is None and 'lat' in d.coords:
                    lats = d.lat.values
                    lons = d.lon.values

                for var in d.data_vars:
                    # Check if percentiles are packed as a dimension
                    if 'percentile' in d.coords:
                        if var in ['t2m', 'tmp', '2t']:
                            try:
                                t_10 = d[var].sel(percentile=10).values
                                t_90 = d[var].sel(percentile=90).values
                            except: pass
                        if var in ['d2m', 'dpt', '2d']:
                            try:
                                td_10 = d[var].sel(percentile=10).values
                                td_90 = d[var].sel(percentile=90).values
                            except: pass
                        if var in ['si10', 'wspd', '10si', 'wind']:
                            try:
                                wind_10 = d[var].sel(percentile=10).values
                                wind_90 = d[var].sel(percentile=90).values
                            except: pass
                            
                    # Check if percentiles are baked into the variable name
                    else:
                        if var in ['t2m_10', 'tmp_10', '2t_10']: t_10 = d[var].values
                        if var in ['t2m_90', 'tmp_90', '2t_90']: t_90 = d[var].values
                        if var in ['d2m_10', 'dpt_10', '2d_10']: td_10 = d[var].values
                        if var in ['d2m_90', 'dpt_90', '2d_90']: td_90 = d[var].values
                        if var in ['si10_10', 'wspd_10', '10si_10']: wind_10 = d[var].values
                        if var in ['si10_90', 'wspd_90', '10si_90']: wind_90 = d[var].values

            if t_10 is None or t_90 is None or td_10 is None or td_90 is None:
                print(f" -> Could not find T/Td percentiles for Hour {fhr}")
                continue
                
            if wind_10 is None or wind_90 is None:
                print(f" -> Could not find Wind percentiles for Hour {fhr}")
                continue

            # --- DERIVE AUTHENTIC PERCENTILES ---
            wind_10 = ms_to_mph(wind_10)
            wind_90 = ms_to_mph(wind_90)

            # Worst-Case RH: Hottest Temp paired with Driest Dewpoint
            rh_10_derived = calculate_rh(t_90, td_10)
            
            # Best-Case RH: Coolest Temp paired with Wettest Dewpoint
            rh_90_derived = calculate_rh(t_10, td_90)

            # --- PROCESS SCENARIOS ---
            worst_case = calculate_fire_danger(rh_10_derived, wind_90, wind_90)
            generate_prob_plot(worst_case, lats, lons, fhr, "worst", "Worst-Case Scenario (Low RH / High Wind)", init_time)
            
            best_case = calculate_fire_danger(rh_90_derived, wind_10, wind_10)
            generate_prob_plot(best_case, lats, lons, fhr, "best", "Best-Case Scenario (High RH / Low Wind)", init_time)
            
            uncertainty = calculate_uncertainty_index(rh_10_derived, rh_90_derived, wind_10, wind_90)
            generate_prob_plot(uncertainty, lats, lons, fhr, "spread", "Forecast Uncertainty Index", init_time, is_uncertainty=True)
            
        except Exception as e:
            print(f"Error processing Hour {fhr}: {e}")
            
        finally:
            for d in datasets:
                try: d.close()
                except: pass
            for junk_file in glob.glob(f"{file_name}*"):
                try: os.remove(junk_file)
                except: pass

if __name__ == "__main__":
    process_nbm()
