import os
import time
import argparse
import logging
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MASTER_CSV = "data/raw/master_dataset.csv"
LOOKUP_CSV = "data/processed/locality_lookup.csv"

def fetch_openmeteo_data(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily precipitation from Open-Meteo Archive API."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "precipitation_sum",
        "timezone": "Asia/Kolkata"
    }
    
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            if "daily" not in data or "time" not in data["daily"]:
                logger.warning(f"No daily data returned for {lat},{lon}")
                return pd.DataFrame()
            
            df = pd.DataFrame({
                "date": data["daily"]["time"],
                "rainfall_mm": data["daily"]["precipitation_sum"]
            })
            df["date"] = pd.to_datetime(df["date"])
            # Drop days where data is not yet available (nulls)
            df = df.dropna(subset=["rainfall_mm"])
            return df
        except Exception as e:
            logger.error(f"API attempt {attempt+1} failed: {e}")
            time.sleep(2 ** attempt)
            
    raise RuntimeError(f"Failed to fetch data for {lat},{lon} after 3 attempts.")

def main(dry_run=False, target_locality=None):
    if not os.path.exists(MASTER_CSV):
        logger.error(f"Cannot find {MASTER_CSV}")
        return
        
    if not os.path.exists(LOOKUP_CSV):
        logger.error(f"Cannot find {LOOKUP_CSV}")
        return
        
    logger.info("Loading datasets...")
    df_master = pd.read_csv(MASTER_CSV)
    df_master["date"] = pd.to_datetime(df_master["date"])
    
    df_lookup = pd.read_csv(LOOKUP_CSV)
    
    today = pd.Timestamp.today().normalize()
    yesterday_str = (today - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    
    new_rows = []
    
    localities = df_lookup["locality"].unique()
    if target_locality:
        localities = [l for l in localities if l == target_locality]
        
    for locality in localities:
        loc_meta = df_lookup[df_lookup["locality"] == locality].iloc[0]
        lat = loc_meta["latitude"]
        lon = loc_meta["longitude"]
        elev = loc_meta["elevation_m_approx"]
        
        # Get existing data for this locality to find max date
        df_loc = df_master[df_master["locality"] == locality].copy()
        df_loc = df_loc.sort_values("date")
        
        if df_loc.empty:
            start_date_str = "2024-01-01"
            logger.warning(f"No existing data for {locality}, starting from {start_date_str}")
        else:
            last_date = df_loc["date"].max()
            start_date = last_date + pd.Timedelta(days=1)
            start_date_str = start_date.strftime("%Y-%m-%d")
            
        if pd.Timestamp(start_date_str) > pd.Timestamp(yesterday_str):
            logger.info(f"[{locality}] Up to date (last: {last_date.date()})")
            continue
            
        logger.info(f"[{locality}] Fetching {start_date_str} to {yesterday_str}...")
        if dry_run:
            continue
            
        # Fetch new data
        try:
            df_new = fetch_openmeteo_data(lat, lon, start_date_str, yesterday_str)
        except Exception as e:
            logger.error(f"Skipping {locality} due to API error: {e}")
            continue
            
        if df_new.empty:
            logger.info(f"[{locality}] No new valid data returned.")
            continue
            
        # Add metadata and derived columns
        df_new["locality"] = locality
        df_new["latitude"] = lat
        df_new["longitude"] = lon
        df_new["elevation_m_approx"] = elev
        
        df_new["year"] = df_new["date"].dt.year
        df_new["month"] = df_new["date"].dt.month
        df_new["day_of_year"] = df_new["date"].dt.dayofyear
        df_new["is_northeast_monsoon"] = df_new["month"].apply(lambda m: 1 if m in [10, 11, 12] else 0)
        df_new["flood_occurred_documented"] = None
        
        # Calculate rolling features
        # We need previous data to seed the rolling windows correctly
        combined = pd.concat([df_loc[["date", "rainfall_mm"]], df_new[["date", "rainfall_mm"]]])
        combined = combined.sort_values("date").set_index("date")
        
        rolling_3d = combined["rainfall_mm"].rolling(window=3, min_periods=1).sum()
        rolling_7d = combined["rainfall_mm"].rolling(window=7, min_periods=1).sum()
        rolling_30d = combined["rainfall_mm"].rolling(window=30, min_periods=1).sum()
        
        df_new = df_new.set_index("date")
        df_new["rainfall_3d_mm"] = rolling_3d.reindex(df_new.index).values
        df_new["rainfall_7d_mm"] = rolling_7d.reindex(df_new.index).values
        df_new["rainfall_30d_mm"] = rolling_30d.reindex(df_new.index).values
        df_new = df_new.reset_index()
        
        # Round numeric columns to match existing format
        numeric_cols = ["rainfall_mm", "rainfall_3d_mm", "rainfall_7d_mm", "rainfall_30d_mm"]
        for c in numeric_cols:
            df_new[c] = df_new[c].round(2)
            
        new_rows.append(df_new)
        # Avoid hitting API rate limits too aggressively
        time.sleep(1)

    if dry_run:
        logger.info("Dry run complete.")
        return

    if not new_rows:
        logger.info("No new data to append.")
        return
        
    df_append = pd.concat(new_rows, ignore_index=True)
    
    # Ensure column order exactly matches the original CSV
    cols = [
        "date", "locality", "latitude", "longitude", "elevation_m_approx",
        "rainfall_mm", "rainfall_3d_mm", "rainfall_7d_mm", "rainfall_30d_mm",
        "year", "month", "day_of_year", "is_northeast_monsoon", "flood_occurred_documented"
    ]
    df_append = df_append[cols]
    
    logger.info(f"Appending {len(df_append)} new rows to {MASTER_CSV}...")
    # Convert date back to string format for consistent CSV saving
    df_append["date"] = df_append["date"].dt.strftime("%Y-%m-%d")
    
    # Append without writing header
    df_append.to_csv(MASTER_CSV, mode="a", index=False, header=False)
    logger.info("Update complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Do not write to CSV or call API")
    parser.add_argument("--locality", type=str, help="Update only a specific locality")
    args = parser.parse_args()
    main(dry_run=args.dry_run, target_locality=args.locality)
