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

# (NBM Data Downloading and Cartopy plotting functions will go here)
