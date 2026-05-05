import os
import glob
import requests
import cfgrib
import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --- 1. CONFIGURATION & THRESHOLDS ---
# NC Extreme (PDS Red Flag)
EXTREME_RH = 20.0
EXTREME_WIND = 25.0
EXTREME_GUST = 35.0

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

# NC Geographic Domain for DSS Max Value Searching
LAT_MIN, LAT_MAX = 32.5, 39.5
LON_MIN, LON_MAX = -85.5, -73.5

# --- GLOBAL DSS SCOREBOARD ---
dss_data = {day: {'ndfd': 0, 'nbm_worst': 0, 'date_str': ''} for day in range(1, 8)}

# --- 2. FIRE DANGER MATH ---
def calculate_fire_danger(rh, wind, gust):
    """
    0 = None (Transparent)
    1 = Low (Yellow)
    2 = Moderate / IFD (Orange)
    3 = High / RFW (Red)
    4 = Extreme (Purple)
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
    
    # 4. Apply Extreme Danger
    extreme_mask = (rh <= EXTREME_RH) & ((wind >= EXTREME_WIND) | (gust >= EXTREME_GUST))
    danger_grid[extreme_mask] = 4
    
    return danger_grid

# --- 3. PLOTTING ENGINE ---
def generate_prob_plot(plot_data, lats, lons, day, scenario, title_text, init_time, fhr):
    fig = plt.figure(figsize=(14, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=1.5, edgecolor='black')
    ax.add_feature(cfeature.COUNTIES.with_scale('5m'), linewidth=0.5, edgecolor='gray')
    
    # 5 colors: Transparent, Bright Yellow, Deep Orange, Crimson Red, Vivid Purple
    cmap = ListedColormap(['#FFFFFF00', '#FFFF00', '#FF6600', '#CC0000', '#9900CC']) 
    levels = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    tick_locs = [0, 1, 2, 3, 4]
    tick_labels = ['None', 'Low', 'Mod', 'High', 'Extreme']

    mesh = ax.pcolormesh(lons, lats, plot_data, cmap=cmap, vmin=levels[0], vmax=levels[-1], transform=ccrs.PlateCarree())
    
    cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.05, aspect=50, ticks=tick_locs)
    cbar.set_ticklabels(tick_labels)
    
    valid_time = init_time + timedelta(hours=fhr)
    plt.title(f"{title_text}\nValid Peak Heating: {valid_time.strftime('%a %m/%d %H:00Z')} (Day {day})", fontsize=14, fontweight='bold')
    
    ax.text(1.03, 0.90, "NWS Raleigh, NC", transform=ax.transAxes, fontsize=12, fontstyle='italic',
            verticalalignment='top', color='#444444')
    
    legend_text = (
        "Threshold Criteria:\n\n"
        "Extreme:\nRH <= 20% AND\n(Wind >= 25 or Gust >= 35 mph)\n\n"
        "High (RFW):\nRH <= 25% AND\n(Wind >= 20 or Gust >= 30 mph)\n\n"
        "Mod (IFD):\nRH <= 30% AND\nGust >= 25 mph\n\n"
        "Low:\nRH <= 35% AND\n(Wind >= 15 or Gust >= 20 mph)"
    )
    
    ax.text(1.03, 0.5, legend_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='center',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#f8f9fa', edgecolor='gray', alpha=0.9))
            
    # Major North Carolina Cities
    cities = {
        'Asheville': (-82.5515, 35.5951),
        'Boone': (-81.6746, 36.2168),
        'Charlotte': (-80.8431, 35.2271),
        'Greensboro': (-79.7922, 36.0726),
        'Raleigh': (-78.6382, 35.7796),
        'Fayetteville': (-78.8784, 35.0527),
        'Greenville': (-77.3664, 35.6127),
        'Wilmington': (-77.9447, 34.2257),
        'Morehead City': (-76.7497, 34.7229),
        'Elizabeth City': (-76.2510, 36.2946),
        'Cape Hatteras': (-75.5200, 35.2672)
    }

    for city, (lon, lat) in cities.items():
        ax.plot(lon, lat, marker='o', color='black', markersize=4, transform=ccrs.PlateCarree())
        ax.text(lon + 0.06, lat + 0.04, city, transform=ccrs.PlateCarree(),
                fontsize=8, fontweight='bold', color='black',
                bbox=dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor='none', alpha=0.6))
    
    os.makedirs('public/images', exist_ok=True)
    temp_filename = f"base_fire_danger_plot.png"
    plt.savefig(temp_filename, bbox_inches='tight', dpi=150)
    plt.close()

    # NWS Logo Composite Logic (Assuming you have nws_logo.png)
    try:
        from PIL import Image
        base_img = Image.open(temp_filename)
        logo = Image.open("nws_logo.png").convert("RGBA")
        logo_height = 120
        aspect_ratio = logo.width / logo.height
        logo_width = int(logo_height * aspect_ratio)
        logo = logo.resize((logo_width, logo_height))
        x_offset = base_img.width - logo_width - 20
        y_offset = 20
        base_img.paste(logo, (x_offset, y_offset), logo)
        final_filename = f"public/images/nbm_{scenario}_fire_danger_day{day}.png"
        base_img.save(final_filename)
        os.remove(temp_filename)
    except Exception as e:
        print(f"Logo overlay failed: {e}")
        os.rename(temp_filename, f"public/images/nbm_{scenario}_fire_danger_day{day}.png")

# --- 4. NBM PROBABILISTIC PROCESSING ---
def process_nbm():
    print("--- Hunting for NBM Grids ---")
    base_url = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/blend/prod"
    now = datetime.utcnow()
    cycle = f"{now.hour//12 * 12:02d}"
    date_str = now.strftime('%Y%m%d')
    url_dir = f"{base_url}/blend.{date_str}/{cycle}/core"
    
    init_time = datetime.strptime(f"{date_str}{cycle}", "%Y%m%d%H")
    
    for day in range(1, 8):
        fhr = day * 24 
        file_name = f"blend.t{cycle}z.core.f{fhr:03d}.co.grib2"
        file_url = f"{url_dir}/{file_name}"
        
        try:
            print(f"Downloading NBM Day {day} (FHR {fhr})...")
            resp = requests.get(file_url, timeout=30)
            if resp.status_code == 200:
                with open(file_name, 'wb') as f:
                    f.write(resp.content)
            else:
                continue

            datasets = cfgrib.open_datasets(file_name)
            
            rh_10 = rh_50 = rh_90 = None
            wind_10 = wind_50 = wind_90 = None
            gust_10 = gust_50 = gust_90 = None
            lats = lons = None
            
            for d in datasets:
                if lats is None and 'latitude' in d.coords:
                    lats = d.latitude.values
                    lons = d.longitude.values
                
                for var in d.data_vars:
                    if 'minrhi' in var:
                        rh_10 = d[var].values if d[var].percentile == 10 else rh_10
                        rh_50 = d[var].values if d[var].percentile == 50 else rh_50
                        rh_90 = d[var].values if d[var].percentile == 90 else rh_90
                    elif '10si' in var and 'wind' in d[var].attrs.get('GRIB_name', '').lower():
                        wind_10 = d[var].values * 2.23694 if d[var].percentile == 10 else wind_10
                        wind_50 = d[var].values * 2.23694 if d[var].percentile == 50 else wind_50
                        wind_90 = d[var].values * 2.23694 if d[var].percentile == 90 else wind_90
                    elif '10si' in var and 'gust' in d[var].attrs.get('GRIB_name', '').lower():
                        gust_10 = d[var].values * 2.23694 if d[var].percentile == 10 else gust_10
                        gust_50 = d[var].values * 2.23694 if d[var].percentile == 50 else gust_50
                        gust_90 = d[var].values * 2.23694 if d[var].percentile == 90 else gust_90

            if None not in [rh_10, wind_90]:
                worst_case = calculate_fire_danger(rh_10, wind_90, gust_90 if gust_90 is not None else wind_90)
                median_case = calculate_fire_danger(rh_50, wind_50, gust_50 if gust_50 is not None else wind_50)
                best_case = calculate_fire_danger(rh_90, wind_10, gust_10 if gust_10 is not None else wind_10)

                generate_prob_plot(worst_case, lats, lons, day, "worst", "NBM Worst-Case Scenario", init_time, fhr)
                generate_prob_plot(median_case, lats, lons, day, "median", "NBM Expected Scenario (Median)", init_time, fhr)
                generate_prob_plot(best_case, lats, lons, day, "best", "NBM Best-Case Scenario", init_time, fhr)

                # --- RECORD NBM WORST-CASE SCORE ---
                valid_time = init_time + timedelta(hours=fhr) 
                lons_180 = np.where(lons > 180, lons - 360, lons)
                nc_mask = (lats >= LAT_MIN) & (lats <= LAT_MAX) & (lons_180 >= LON_MIN) & (lons_180 <= LON_MAX)
                
                dss_data[day]['nbm_worst'] = int(np.max(worst_case[nc_mask]))
                dss_data[day]['date_str'] = valid_time.strftime('%A, %b %d')

        except Exception as e:
            print(f"Error processing NBM Day {day}: {e}")
            
        finally:
            for d in datasets:
                try: d.close()
                except: pass
            for junk in glob.glob(f"{file_name}*"):
                try: os.remove(junk)
                except: pass

# --- 5. NDFD OFFICIAL FORECAST PROCESSING ---
def process_ndfd():
    print("--- Hunting for Official NWS NDFD Grids ---")
    base_url = "https://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/DF.gr2/DC.ndfd/AR.conus"
    periods = ["VP.001-003", "VP.004-007"]
    variables = ["ds.minrh.bin", "ds.wspd.bin", "ds.wgust.bin"]
    now_local = datetime.now(ZoneInfo("America/New_York"))
    
    all_datasets = {'minrh': [], 'wspd': [], 'wgust': []}
    lats = lons = None
    
    for period in periods:
        for var_file in variables:
            file_url = f"{base_url}/{period}/{var_file}"
            local_file = f"ndfd_{period}_{var_file}"
            try:
                resp = requests.get(file_url, timeout=30)
                if resp.status_code == 200:
                    with open(local_file, 'wb') as f:
                        f.write(resp.content)
                    datasets = cfgrib.open_datasets(local_file)
                    var_key = var_file.split('.')[1]
                    for d in datasets:
                        if lats is None and 'latitude' in d.coords:
                            lats = d.latitude.values
                            lons = d.longitude.values
                        all_datasets[var_key].append(d)
            except Exception as e:
                pass 

    for day in range(1, 8):
        target_date = (now_local + timedelta(days=day-1)).date()
        window_start = datetime(target_date.year, target_date.month, target_date.day, 12, 0, tzinfo=timezone.utc)
        window_end = window_start + timedelta(hours=18)
        
        daily_rh_list, daily_wind_list, daily_gust_list = [], [], []
        print(f"Processing NDFD Official Forecast for Day {day} ({target_date})...")
        
        try:
            for var_key, ds_list in all_datasets.items():
                for d in ds_list:
                    for var in d.data_vars:
                        da = d[var]
                        if 'valid_time' in da.coords:
                            vtimes = da.valid_time.values
                            if vtimes.ndim == 0:
                                vtimes = [vtimes]
                                vals = np.expand_dims(da.values, axis=0)
                            else:
                                vals = da.values
                                
                            for i, vt in enumerate(vtimes):
                                vt_utc = pd.to_datetime(vt).tz_localize('UTC')
                                if window_start <= vt_utc <= window_end:
                                    if var_key == 'minrh': daily_rh_list.append(vals[i])
                                    elif var_key == 'wspd': daily_wind_list.append(vals[i])
                                    elif var_key == 'wgust': daily_gust_list.append(vals[i])

            if daily_rh_list and daily_wind_list:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    daily_rh = np.nanmin(np.array(daily_rh_list), axis=0)
                    daily_wind = np.nanmax(np.array(daily_wind_list), axis=0)
                    if daily_gust_list: daily_gust = np.nanmax(np.array(daily_gust_list), axis=0)
                    else: daily_gust = daily_wind
                        
                daily_wind_mph = daily_wind * 2.23694
                daily_gust_mph = daily_gust * 2.23694
                official_case = calculate_fire_danger(daily_rh, daily_wind_mph, daily_gust_mph)
                
                # --- RECORD NDFD SCORE ---
                lons_180 = np.where(lons > 180, lons - 360, lons)
                nc_mask = (lats >= LAT_MIN) & (lats <= LAT_MAX) & (lons_180 >= LON_MIN) & (lons_180 <= LON_MAX)
                dss_data[day]['ndfd'] = int(np.max(official_case[nc_mask]))
                
                plot_time_utc = datetime(target_date.year, target_date.month, target_date.day, 21, 0)
                generate_prob_plot(official_case, lats, lons, day, "official", "Official NWS Forecast (NDFD)", plot_time_utc, 0)
            else:
                print(f" -> No daytime data left on server for Day {day}. Generating safe blank map.")
                dss_data[day]['ndfd'] = 0 # Force score to 0 for expired maps
                if lats is not None:
                    dummy_rh = np.ones_like(lats) * 100 
                    dummy_wind = np.zeros_like(lats)    
                    empty_case = calculate_fire_danger(dummy_rh, dummy_wind, dummy_wind)
                    plot_time_utc = datetime(target_date.year, target_date.month, target_date.day, 21, 0)
                    generate_prob_plot(empty_case, lats, lons, day, "official", "Official Forecast (NDFD) - DAY EXPIRED", plot_time_utc, 0)
                
        except Exception as e:
            print(f"Error processing NDFD Day {day}: {e}")

    for ds_list in all_datasets.values():
        for d in ds_list:
            try: d.close()
            except: pass
    for junk in glob.glob("ndfd_*"):
        try: os.remove(junk)
        except: pass

# --- 6. GENERATE FINAL DSS BULLETIN ---
def generate_dss_bulletin():
    print("--- Generating Unified DSS Text Bulletin ---")
    dss_lines = []
    
    for day in range(1, 8):
        ndfd_lvl = dss_data[day]['ndfd']
        worst_lvl = dss_data[day]['nbm_worst']
        day_name = dss_data[day]['date_str']
        
        if not day_name: continue 
        if ndfd_lvl == 0 and worst_lvl == 0: continue 
        
        if ndfd_lvl == 4:
            status = f"<strong>Day {day} ({day_name}): EXTREME (Official).</strong> Official NWS forecast expects extreme criteria with critically low RH and damaging winds."
        elif ndfd_lvl == 3:
            status = f"<strong>Day {day} ({day_name}): High / RFW (Official).</strong> Official NWS forecast reaches Red Flag Warning criteria."
            if worst_lvl == 4: status += " <em>Note: The worst-case scenario shows localized EXTREME fire behavior is possible.</em>"
        elif ndfd_lvl == 2:
            status = f"<strong>Day {day} ({day_name}): Mod / IFD (Official).</strong> Official NWS forecast reaches Increased Fire Danger criteria."
            if worst_lvl >= 3: status += " <em>Note: The worst-case scenario shows potential for localized Red Flag conditions if winds overperform.</em>"
        elif ndfd_lvl == 1:
            status = f"<strong>Day {day} ({day_name}): Low (Official).</strong> Breezy and dry conditions expected, but official forecast remains below IFD thresholds."
            if worst_lvl >= 2: status += " <em>Note: The worst-case scenario indicates IFD or Red Flag conditions cannot be completely ruled out.</em>"
        else:
            status = f"<strong>Day {day} ({day_name}): Expected None.</strong> Official forecast remains below elevated thresholds."
            if worst_lvl >= 2: status += " <em>Note: The worst-case scenario indicates localized elevated conditions (IFD) cannot be entirely ruled out.</em>"
            elif worst_lvl == 1: status += " <em>Note: The worst-case scenario indicates localized Low fire danger is possible.</em>"
            
        dss_lines.append(f"<li>{status}</li>")

    now_time = datetime.now(ZoneInfo("America/New_York")).strftime('%A, %B %d, %Y at %I:%M %p %Z')
    
    with open('public/timestamp.txt', 'w') as f:
        f.write(f"Last Refreshed: {now_time}")
        
    with open('public/dss_bulletin.html', 'w') as f:
        f.write(f"<p style='color: #0056b3; font-weight: bold; text-align: left; margin-top: 0; border-bottom: 1px solid #ddd; padding-bottom: 8px;'>Data Last Refreshed: {now_time}</p>\n")
        
        if len(dss_lines) == 0:
            f.write("<p style='text-align: left; font-weight: bold; color: #2e7d32; padding: 10px 0;'>No elevated fire weather threats expected over the next 7 days.</p>\n")
        else:
            f.write("<ul style='text-align: left; line-height: 1.6;'>\n" + "\n".join(dss_lines) + "\n</ul>\n")
            
        f.write("<p style='font-size: 12px; color: gray; text-align: left; margin-top: 15px;'><em>*Disclaimer: This automated guidance evaluates meteorological conditions only and does not account for local fuel moisture (ERC). Consult official NWS forecasts for operational decisions.</em></p>")

if __name__ == "__main__":
    process_nbm()
    process_ndfd()
    generate_dss_bulletin()
