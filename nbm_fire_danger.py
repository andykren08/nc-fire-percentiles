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
# NC Extreme (PDS Red Flag)
EXTREME_RH = 25.0
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

lat_min, lat_max = 32.5, 39.5
lon_min, lon_max = -85.5, -73.5

os.makedirs('public/images', exist_ok=True)

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

def ms_to_mph(ms):
    return ms * 2.23694

# --- 3. MAPPING ---
def generate_prob_plot(plot_data, lats, lons, day, scenario, title_text, init_time, fhr):
    # Import PIL for image compositing (must be imported within the function)
    from PIL import Image
    import PIL.ImageOps

    fig = plt.figure(figsize=(14, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.STATES, linewidth=1.5, edgecolor='black')
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0)
    
    try:
        from metpy.plots import USCOUNTIES
        ax.add_feature(USCOUNTIES.with_scale('5m'), edgecolor='gray', linewidth=0.5)
    except: pass

    from matplotlib.colors import ListedColormap
    # 5 colors: Transparent, Bright Yellow, Deep Orange, Crimson Red, Vivid Purple
    cmap = ListedColormap(['#FFFFFF00', '#FFFF00', '#FF6600', '#CC0000', '#9900CC']) 
    levels = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    tick_locs = [0, 1, 2, 3, 4]
    tick_labels = ['None', 'Low', 'Mod', 'High', 'Extreme']

    mesh = ax.pcolormesh(lons, lats, plot_data, cmap=cmap, vmin=levels[0], vmax=levels[-1], transform=ccrs.PlateCarree())
    
    cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.05, aspect=50, ticks=tick_locs)
    cbar.set_ticklabels(tick_labels)
    
    valid_time = init_time + timedelta(hours=fhr)
    plt.title(f" {title_text}\nValid Peak Heating: {valid_time.strftime('%a %m/%d %H:00Z')} (Day {day})", fontsize=14, fontweight='bold')
    
    # Add NWS Raleigh text under the logo ---
    ax.text(1.13, 0.83, "NWS Raleigh, NC", transform=ax.transAxes, fontsize=12, fontstyle='italic',
            verticalalignment='top', color='#444444')
    
    # Burn the threshold legend directly into the image
    legend_text = (
        "Threshold Criteria:\n\n"
        "Extreme:\nRH <= 20% AND\n(Wind >= 25 or Gust >= 35 mph)\n\n"
        "High (Red Flag):\nRH <= 25% AND\n(Wind >= 20 or Gust >= 30 mph)\n\n"
        "Mod (IFD):\nRH <= 30% AND\nGust >= 25 mph\n\n"
        "Low:\nRH <= 35% AND\n(Wind >= 15 or Gust >= 20 mph)"
    )
    
    # Place text box slightly outside the right edge of the map
    ax.text(1.03, 0.5, legend_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='center',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#f8f9fa', edgecolor='gray', alpha=0.9))
    
    # Define filenames for the intermediate and final images
    temp_filename = f"base_fire_danger_plot.png"
    final_filename = f"public/images/nbm_{scenario}_fire_danger_day{day}.png"

    # Step 1: Save the initial plot with all its legends.
    # 'bbox_inches=tight' handles the dynamic text legend width.

    # Place text box slightly outside the right edge of the map
    ax.text(1.03, 0.5, legend_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='center',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#f8f9fa', edgecolor='gray', alpha=0.9))
            
    # --- NEW: Plot Major North Carolina Cities ---
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
        # Plot a small black dot for the city location
        ax.plot(lon, lat, marker='o', color='black', markersize=4, transform=ccrs.PlateCarree())
        
        # Add the city text label slightly offset from the dot
        ax.text(lon + 0.06, lat + 0.04, city, transform=ccrs.PlateCarree(),
                fontsize=8, fontweight='bold', color='black',
                bbox=dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor='none', alpha=0.6))
    
    # Define filenames for the intermediate and final images
    temp_filename = f"base_fire_danger_plot.png"
    
    plt.savefig(temp_filename, bbox_inches='tight', dpi=150)
    plt.close()

    # Step 2: Composite the logo after the plot is rendered.
    try:
        # Load the saved plot and the logo
        base_image = Image.open(temp_filename)
        # Using the filename of the provided logo
        logo = Image.open('image_0.png')
        logo = logo.convert("RGBA") # Ensure alpha transparency is correct

        # Scale the logo. Let's aim for ~200 pixels tall to match the title.
        logo_height = 200
        aspect_ratio = logo.size[0] / logo.size[1]
        logo_width = int(logo_height * aspect_ratio)
        scaled_logo = logo.resize((logo_width, logo_height))

        # Calculate position for top-right of the visual plot area.
        # base_image.size gives the width and height of the saved plot.
        # We place it with a 20px padding from the top edge.
        logo_x = base_image.size[0] - scaled_logo.size[0] - 20
        logo_y = 20 

        # Paste the scaled logo over the base image, using the alpha channel as a mask
        base_image.paste(scaled_logo, (logo_x, logo_y), scaled_logo)

        # Step 3: Save the final composited image to its official location.
        base_image.save(final_filename)

    except FileNotFoundError:
        print(f"CRITICAL: Logo file (image_0.png) not found in directory.")
        # Fallback to saving without logo if needed
        os.rename(temp_filename, final_filename)
    except Exception as e:
        print(f"Error compositing logo: {e}")
        os.rename(temp_filename, final_filename)
    finally:
        # Cleanup the temporary base image
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

# --- 4. MAIN PROCESSING LOOP (NBM TRUE PROBABILISTIC QMD) ---
def process_nbm():
    import subprocess
    import xarray as xr
    print("--- Hunting for NBM QMD Cycles (True Probabilistic Forecast) ---")
    now = datetime.utcnow()
    valid_cycle_found = False
    
    for hours_back in range(0, 48):
        check_time = now - timedelta(hours=hours_back)
        cycle_hour = (check_time.hour // 6) * 6 
        
        date_str = check_time.strftime("%Y%m%d")
        hour_str = f"{cycle_hour:02d}"
        
        test_url = f"https://noaa-nbm-grib2-pds.s3.amazonaws.com/blend.{date_str}/{hour_str}/qmd/blend.t{hour_str}z.qmd.f012.co.grib2.idx"
        
        try:
            if requests.head(test_url, timeout=5).status_code == 200:
                print(f"Success! Locked onto fresh NBM QMD Cycle: {date_str} at {hour_str}Z")
                valid_cycle_found = True
                break
        except: pass 

    if not valid_cycle_found:
        print("CRITICAL: Could not find any uploaded NBM QMD cycles.")
        return

    init_time = datetime(check_time.year, check_time.month, check_time.day, cycle_hour, 0)
    
    base_fhr = 21 - cycle_hour
    if base_fhr < 0: base_fhr += 24
    
    nomads_left = 360 + lon_min - 1.0
    nomads_right = 360 + lon_max + 1.0
    nomads_top = lat_max + 1.0
    nomads_bottom = lat_min - 1.0
    
    for day in range(1, 8):
        fhr = base_fhr + (day - 1) * 24
        file_name = f"blend.t{hour_str}z.qmd.f{fhr:03d}.co.grib2"
        
       # FIXED FILTER: Grab all variables, but restrict it geographically to NC!
        filter_url = (
            f"https://nomads.ncep.noaa.gov/cgi-bin/filter_blend.pl"
            f"?file={file_name}&all_lev=on&all_var=on"
            f"&subregion=on&leftlon={nomads_left}&rightlon={nomads_right}"
            f"&toplat={nomads_top}&bottomlat={nomads_bottom}"
            f"&dir=%2Fblend.{date_str}%2F{hour_str}%2Fqmd"
        )
        
        try:
            print(f"Downloading NBM QMD Day {day} (F{fhr:03d}) via NOMADS grib_filter...")
            resp = requests.get(filter_url, timeout=60)
            
            # If the file is too small, it's an HTML error page from NOMADS
            if resp.status_code != 200 or len(resp.content) < 1000: 
                print(f" -> Day {day} unavailable on NOMADS or filter blocked.")
                print(f" -> NOMADS Message: {resp.text[:250]}") # Shows us the exact error!
                continue
                
            with open(file_name, 'wb') as f:
                f.write(resp.content)
            
            # Use WGRIB2 to build the NetCDF
            nc_file = f"{file_name}.nc"
            subprocess.run(['wgrib2', file_name, '-netcdf', nc_file], check=True, capture_output=True)
            
            ds = xr.open_dataset(nc_file)
            
            rh_10 = rh_50 = rh_90 = None
            wind_10 = wind_50 = wind_90 = None
            gust_10 = gust_50 = gust_90 = None
            
            lats = ds.latitude.values
            lons = ds.longitude.values

            # --- THE SMART VARIABLE SCRUBBER ---
            for var in ds.data_vars:
                da = ds[var]
                var_name = str(var).lower()
                desc = str(da.attrs.get('long_name', '')).lower()
                
                # Skip basic coordinates
                if var_name in ['lat', 'lon', 'latitude', 'longitude', 'time']: continue
                
                val = da.values
                if val.ndim >= 3: val = val[0] 
                if val.ndim >= 3: val = val[0] 
                
                # Scrub out height indicators so '10m' wind doesn't falsely trigger the 10% logic
                search_str = f"{var_name} {desc}".replace('10m', '').replace('10 m', '').replace('2m', '').replace('2 m', '')
                
                # RH Parser
                if 'rh' in var_name or 'relative humidity' in desc:
                    if '10' in search_str: rh_10 = val
                    elif '50' in search_str: rh_50 = val
                    elif '90' in search_str: rh_90 = val
                
                # Wind Parser
                elif 'wind' in var_name or 'wspd' in var_name:
                    if '10' in search_str: wind_10 = val * 2.23694
                    elif '50' in search_str: wind_50 = val * 2.23694
                    elif '90' in search_str: wind_90 = val * 2.23694
                
                # Gust Parser
                elif 'gust' in var_name:
                    if '10' in search_str: gust_10 = val * 2.23694
                    elif '50' in search_str: gust_50 = val * 2.23694
                    elif '90' in search_str: gust_90 = val * 2.23694

            # --- DIAGNOSTIC DEBUG TOOL ---
            if None in [rh_10, wind_90]:
                print(f" -> Missing required QMD percentile data for Day {day}")
                print("DEBUG: Here are the exact variable formats generated by wgrib2:")
                for v in ds.data_vars:
                    print(f"  - {v}: {ds[v].dims} | {ds[v].attrs.get('long_name', '')}")
                ds.close()
                continue

            ds.close() 

            if gust_90 is None: gust_90 = wind_90
            if gust_50 is None: gust_50 = wind_50
            if gust_10 is None: gust_10 = wind_10

            worst_case = calculate_fire_danger(rh_10, wind_90, gust_90)
            generate_prob_plot(worst_case, lats, lons, day, "worst", "NBM Worst-Case (10% RH / 90% Wind)", init_time, fhr)
            
            median_case = calculate_fire_danger(rh_50, wind_50, gust_50)
            generate_prob_plot(median_case, lats, lons, day, "median", "NBM Expected (50th Percentile)", init_time, fhr)

            best_case = calculate_fire_danger(rh_90, wind_10, gust_10)
            generate_prob_plot(best_case, lats, lons, day, "best", "NBM Best-Case (90% RH / 10% Wind)", init_time, fhr)

            valid_time = init_time + timedelta(hours=fhr) 
            lons_180 = np.where(lons > 180, lons - 360, lons)
            nc_mask = (lats >= lat_min) & (lats <= lat_max) & (lons_180 >= lon_min) & (lons_180 <= lon_max)
            
            dss_data[day]['nbm_worst'] = int(np.max(worst_case[nc_mask]))
            dss_data[day]['date_str'] = valid_time.strftime('%A, %b %d')

        except subprocess.CalledProcessError as e:
            print(f"wgrib2 conversion failed for Day {day}: {e.stderr.decode()}")
        except Exception as e:
            print(f"Error processing NBM QMD Day {day}: {e}")
            
        finally:
            for junk in glob.glob(f"{file_name}*"):
                try: os.remove(junk)
                except: pass

# --- 5. NDFD OFFICIAL FORECAST PROCESSING ---
def process_ndfd():
    import warnings
    import pandas as pd
    from zoneinfo import ZoneInfo
    from datetime import datetime, timedelta, timezone
    
    print("--- Hunting for Official NWS NDFD Grids ---")
    
    base_url = "https://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/DF.gr2/DC.ndfd/AR.conus"
    periods = ["VP.001-003", "VP.004-007"]
    variables = ["ds.minrh.bin", "ds.wspd.bin", "ds.wgust.bin"]
    
    now_local = datetime.now(ZoneInfo("America/New_York"))
    
    # STEP 1: Download ALL files first and dump them into a bucket to avoid file boundary gaps!
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
                    var_key = var_file.split('.')[1] # Extracts 'minrh', 'wspd', or 'wgust'
                    
                    for d in datasets:
                        if lats is None and 'latitude' in d.coords:
                            lats = d.latitude.values
                            lons = d.longitude.values
                        all_datasets[var_key].append(d)
            except Exception as e:
                pass # Skip silently if a specific slice is unavailable

    # STEP 2: Loop through the 7 days and extract by strict UTC time window
    for day in range(1, 8):
        target_date = (now_local + timedelta(days=day-1)).date()
        
        # We only want grids valid during the "Peak Heating" window for the target date:
        # 12Z (8 AM EDT) through 06Z the next morning (2 AM EDT). 
        # This catches the afternoon winds and the 00Z Min RH grid perfectly!
        window_start = datetime(target_date.year, target_date.month, target_date.day, 12, 0, tzinfo=timezone.utc)
        window_end = window_start + timedelta(hours=18)
        
        daily_rh_list = []
        daily_wind_list = []
        daily_gust_list = []
        
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
                                # Convert GRIB valid time to UTC timezone-aware datetime
                                vt_utc = pd.to_datetime(vt).tz_localize('UTC')
                                
                                # ISOLATE: Does this grid fall inside our daytime window?
                                if window_start <= vt_utc <= window_end:
                                    if var_key == 'minrh': daily_rh_list.append(vals[i])
                                    elif var_key == 'wspd': daily_wind_list.append(vals[i])
                                    elif var_key == 'wgust': daily_gust_list.append(vals[i])

            # Math & Plotting
            if daily_rh_list and daily_wind_list:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    daily_rh = np.nanmin(np.array(daily_rh_list), axis=0)
                    daily_wind = np.nanmax(np.array(daily_wind_list), axis=0)
                    
                    if daily_gust_list:
                        daily_gust = np.nanmax(np.array(daily_gust_list), axis=0)
                    else:
                        daily_gust = daily_wind
                        
                daily_wind_mph = daily_wind * 2.23694
                daily_gust_mph = daily_gust * 2.23694
                
                official_case = calculate_fire_danger(daily_rh, daily_wind_mph, daily_gust_mph)
                plot_time_utc = datetime(target_date.year, target_date.month, target_date.day, 21, 0)
                generate_prob_plot(official_case, lats, lons, day, "official", "Official NWS Forecast (NDFD)", plot_time_utc, 0)
                
            else:
                # --- NEW: ANTI-CRASH FALLBACK ---
                # Generate a safe, blank map so the HTML dropdown doesn't show a broken image!
                print(f" -> No daytime data left on server for Day {day}. Generating safe blank map.")
                if lats is not None:
                    dummy_rh = np.ones_like(lats) * 100  # Force 100% RH
                    dummy_wind = np.zeros_like(lats)     # Force 0 mph Wind
                    empty_case = calculate_fire_danger(dummy_rh, dummy_wind, dummy_wind)
                    
                    plot_time_utc = datetime(target_date.year, target_date.month, target_date.day, 21, 0)
                    generate_prob_plot(empty_case, lats, lons, day, "official", "Official Forecast (NDFD) - DAY EXPIRED", plot_time_utc, 0)
                
        except Exception as e:
            print(f"Error processing NDFD Day {day}: {e}")

    # STEP 3: Clean up all files and memory
    for ds_list in all_datasets.values():
        for d in ds_list:
            try: d.close()
            except: pass
    for junk in glob.glob("ndfd_*"):
        try: os.remove(junk)
        except: pass

if __name__ == "__main__":
    process_nbm()
    process_ndfd()
