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
# NC High (Red Flag Warning)
HIGH_RH = 25.0
HIGH_WIND = 20.0
HIGH_GUST = 30.0

# NC Moderate (Increased Fire Danger)
MOD_RH = 30.0
MOD_GUST = 25.0

# NC Low (Early Heads-Up)
LOW_RH = 35.0
LOW_WIND = 15.0
LOW_GUST = 20.0

lat_min, lat_max = 32.5, 39.5
lon_min, lon_max = -85.5, -73.5

os.makedirs('public/images', exist_ok=True)

# --- 2. FIRE DANGER MATH ---
def calculate_fire_danger(rh, wind, gust):
    """
    0 = None (Green)
    1 = Low (Yellow)
    2 = Moderate / IFD (Orange)
    3 = High / RFW (Red)
    """
    danger_grid = np.zeros_like(rh, dtype=int)
    
    # 1. Apply Low Danger
    low_mask = (rh <= LOW_RH) & ((wind >= LOW_WIND) | (gust >= LOW_GUST))
    danger_grid[low_mask] = 1
    
    # 2. Apply Moderate (IFD) Danger
    mod_mask = (rh <= MOD_RH) & (gust >= MOD_GUST)
    danger_grid[mod_mask] = 2
    
    # 3. Apply High (RFW) Danger
    high_mask = (rh <= HIGH_RH) & ((wind >= HIGH_WIND) | (gust >= HIGH_GUST))
    danger_grid[high_mask] = 3
    
    return danger_grid

def ms_to_mph(ms):
    return ms * 2.23694

# --- 3. MAPPING ---
def generate_prob_plot(plot_data, lats, lons, day, scenario, title_text, init_time, fhr):
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.STATES, linewidth=1.5, edgecolor='black')
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0)
    
    try:
        from metpy.plots import USCOUNTIES
        ax.add_feature(USCOUNTIES.with_scale('5m'), edgecolor='gray', linewidth=0.5)
    except: pass

    from matplotlib.colors import ListedColormap
    # 4 colors: Green, Yellow, Orange, Red
    cmap = ListedColormap(['#A1D99B', '#FEE08B', '#FDAE61', '#D73027']) 
    levels = [-0.5, 0.5, 1.5, 2.5, 3.5]
    tick_locs = [0, 1, 2, 3]
    tick_labels = ['None', 'Low', 'Mod (IFD)', 'High (RFW)']

    mesh = ax.pcolormesh(lons, lats, plot_data, cmap=cmap, vmin=levels[0], vmax=levels[-1], transform=ccrs.PlateCarree())
    
    cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.05, aspect=50, ticks=tick_locs)
    cbar.set_ticklabels(tick_labels)
    
    valid_time = init_time + timedelta(hours=fhr)
    plt.title(f"NBM {title_text}\nValid Peak Heating: {valid_time.strftime('%a %m/%d %H:00Z')} (Day {day})", fontsize=14, fontweight='bold')
    
    # --- NEW: Burn the threshold legend directly into the image ---
    legend_text = (
        "Threshold Criteria:\n\n"
        "High (RFW):\nRH <= 25% AND\n(Wind >= 20 or Gust >= 30 mph)\n\n"
        "Mod (IFD):\nRH <= 30% AND\nGust >= 25 mph\n\n"
        "Low:\nRH <= 35% AND\n(Wind >= 15 or Gust >= 20 mph)"
    )
    
    # Place text box slightly outside the right edge of the map
    ax.text(1.03, 0.5, legend_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='center',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#f8f9fa', edgecolor='gray', alpha=0.9))
    
    # 'bbox_inches=tight' automatically expands the saved image to fit the new text box
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
    
    base_fhr = 21 - cycle_hour
    if base_fhr < 0: base_fhr += 24
    
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

            if gust_det is None: gust_det = wind_det

            wind_det = ms_to_mph(wind_det)
            gust_det = ms_to_mph(gust_det)

            # SIMULATE PROPORTIONAL SPREAD
            rh_10 = np.clip(rh_det * 0.8, 5, 100) 
            rh_90 = np.clip(rh_det * 1.2, 5, 100) 
            
            wind_10 = np.clip(wind_det * 0.6, 0, 100)
            wind_90 = wind_det * 1.4                  
            gust_90 = gust_det * 1.4

            # 1. Worst-Case
            worst_case = calculate_fire_danger(rh_10, wind_90, gust_90)
            generate_prob_plot(worst_case, lats, lons, day, "worst", "Worst-Case Scenario (Low RH / High Wind)", init_time, fhr)
            
            # 2. Expected (Median)
            median_case = calculate_fire_danger(rh_det, wind_det, gust_det)
            generate_prob_plot(median_case, lats, lons, day, "median", "Expected Scenario (Median Forecast)", init_time, fhr)

            # 3. Best-Case
            best_case = calculate_fire_danger(rh_90, wind_10, wind_10)
            generate_prob_plot(best_case, lats, lons, day, "best", "Best-Case Scenario (High RH / Low Wind)", init_time, fhr)
            
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
