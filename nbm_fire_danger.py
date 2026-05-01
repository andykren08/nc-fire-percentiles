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

lat_min, lat_max = 32.5, 39.5
lon_min, lon_max = -85.5, -73.5

os.makedirs('public/images', exist_ok=True)

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

    dss_lines = []
    
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

        # --- AUTOMATED DSS BULLETIN LOGIC ---
            valid_time = init_time + timedelta(hours=fhr) 
            
            lons_180 = np.where(lons > 180, lons - 360, lons)
            nc_mask = (lats >= lat_min) & (lats <= lat_max) & (lons_180 >= lon_min) & (lons_180 <= lon_max)
            
            max_median = np.max(median_case[nc_mask])
            max_worst = np.max(worst_case[nc_mask])
            
            day_name = valid_time.strftime('%A, %b %d')

            # EXCEPTION-BASED REPORTING: Only generate text if there is a threat
            if max_median == 0 and max_worst == 0:
                pass # Completely green day, skip adding it to the bulletin
            else:
                if max_median == 4:
                    status = f"<strong>Day {day} ({day_name}): EXTREME (PDS Red Flag Threat).</strong> Expected forecast reaches extreme criteria with critically low RH and gusty winds."
                elif max_median == 3:
                    status = f"<strong>Day {day} ({day_name}): High (Red Flag Threat).</strong> Expected forecast reaches RFW criteria during peak heating."
                    if max_worst == 4:
                        status += " <em>Note: The worst-case scenario shows localized EXTREME fire behavior is possible if winds overperform.</em>"
                elif max_median == 2:
                    status = f"<strong>Day {day} ({day_name}): Mod (IFD).</strong> Expected forecast reaches Increased Fire Danger criteria."
                    if max_worst >= 3:
                        status += " <em>Note: The worst-case scenario shows localized Red Flag conditions are possible if the environment trends drier/windier.</em>"
                elif max_median == 1:
                    status = f"<strong>Day {day} ({day_name}): Low.</strong> Breezy and dry conditions possible, but generally remaining below IFD thresholds."
                    if max_worst >= 2:
                        status += " <em>Note: The worst-case scenario shows IFD conditions cannot be completely ruled out.</em>"
                else:
                    # Handles cases where median is Green, but Worst-Case implies a threat
                    status = f"<strong>Day {day} ({day_name}): None.</strong> Peak heating expected conditions are below thresholds."
                    if max_worst >= 1:
                        status += " <em>Note: The worst-case scenario indicates localized Low/Elevated conditions cannot be entirely ruled out.</em>"
                
                dss_lines.append(f"<li>{status}</li>")
            
        except Exception as e:
            print(f"Error processing Day {day}: {e}")
            
        finally:
            for d in datasets:
                try: d.close()
                except: pass
            for junk in glob.glob(f"{file_name}*"):
                try: os.remove(junk)
                except: pass

    # --- SAVE THE DSS BULLETIN WITH TIMESTAMP ---
    from zoneinfo import ZoneInfo
    # Fetch the exact time the Action runs, localized to Eastern Time
    now_time = datetime.now(ZoneInfo("America/New_York")).strftime('%A, %B %d, %Y at %I:%M %p %Z')
    
    # NEW: Save just the timestamp to a text file for the top of the website
    with open('public/timestamp.txt', 'w') as f:
        f.write(f"Last Refreshed: {now_time}")
    
    with open('public/dss_bulletin.html', 'w') as f:
        # Add the Timestamp to the top of the DSS box
        f.write(f"<p style='color: #0056b3; font-weight: bold; text-align: left; margin-top: 0; border-bottom: 1px solid #ddd; padding-bottom: 8px;'>Data Last Refreshed: {now_time}</p>\n")
        
        # If all 7 days were green, output an "All Clear" message
        if len(dss_lines) == 0:
            f.write("<p style='text-align: left; font-weight: bold; color: #2e7d32; padding: 10px 0;'>No elevated fire weather threats expected over the next 7 days.</p>\n")
        else:
            f.write("<ul style='text-align: left; line-height: 1.6;'>\n" + "\n".join(dss_lines) + "\n</ul>\n")
            
        f.write("<p style='font-size: 12px; color: gray; text-align: left; margin-top: 15px;'><em>*Disclaimer: This automated guidance evaluates meteorological conditions only and does not account for local fuel moisture (ERC). Consult official NWS forecasts for operational decisions.</em></p>")

# --- 5. NDFD OFFICIAL FORECAST PROCESSING ---
def process_ndfd():
    import warnings
    print("--- Hunting for Official NWS NDFD Grids ---")
    
    base_url = "https://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/DF.gr2/DC.ndfd/AR.conus"
    periods = ["VP.001-003", "VP.004-007"]
    variables = ["ds.minrh.bin", "ds.wspd.bin", "ds.wgst.bin"]
    
    from zoneinfo import ZoneInfo
    init_time = datetime.now(ZoneInfo("UTC"))
    
    for day in range(1, 8):
        period = periods[0] if day <= 3 else periods[1]
        
        daily_rh = None
        daily_wind = None
        daily_gust = None
        lats = lons = None
        
        try:
            print(f"Processing NDFD Official Forecast for Day {day}...")
            
            for var_file in variables:
                file_url = f"{base_url}/{period}/{var_file}"
                local_file = f"ndfd_{var_file}"
                
                resp = requests.get(file_url, timeout=30)
                if resp.status_code == 200:
                    with open(local_file, 'wb') as f:
                        f.write(resp.content)
                else:
                    continue
                
                datasets = cfgrib.open_datasets(local_file)
                for d in datasets:
                    if lats is None and 'latitude' in d.coords:
                        lats = d.latitude.values
                        lons = d.longitude.values
                        
                    for var in d.data_vars:
                        val = d[var].values
                        
                        # Temporarily suppress warnings for all-NaN ocean slices
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=RuntimeWarning)
                            
                            # Use np.nanmin and np.nanmax to ignore missing data!
                            if 'minrh' in var_file:
                                min_val = np.nanmin(val, axis=0) if val.ndim > 2 else val
                                if daily_rh is None: daily_rh = min_val
                                else: daily_rh = np.fmin(daily_rh, min_val)
                                
                            elif 'wspd' in var_file:
                                max_val = np.nanmax(val, axis=0) if val.ndim > 2 else val
                                if daily_wind is None: daily_wind = max_val
                                else: daily_wind = np.fmax(daily_wind, max_val)
                                
                            elif 'wgst' in var_file:
                                max_val = np.nanmax(val, axis=0) if val.ndim > 2 else val
                                if daily_gust is None: daily_gust = max_val
                                else: daily_gust = np.fmax(daily_gust, max_val)

                for d in datasets:
                    try: d.close()
                    except: pass
                for junk in glob.glob(f"{local_file}*"):
                    try: os.remove(junk)
                    except: pass

            if daily_rh is not None and daily_wind is not None:
                # FIX: NDFD winds are strictly m/s! Convert to mph properly.
                daily_wind_mph = daily_wind * 2.23694
                if daily_gust is not None:
                    daily_gust_mph = daily_gust * 2.23694
                else:
                    daily_gust_mph = daily_wind_mph
                
                official_case = calculate_fire_danger(daily_rh, daily_wind_mph, daily_gust_mph)
                generate_prob_plot(official_case, lats, lons, day, "official", "Official NWS Forecast (NDFD)", init_time, day * 24)
                
        except Exception as e:
            print(f"Error processing NDFD Day {day}: {e}")

if __name__ == "__main__":
    process_nbm()
    process_ndfd()
