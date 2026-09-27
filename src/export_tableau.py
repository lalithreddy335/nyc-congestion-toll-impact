"""
Build every data file the Tableau dashboard needs, into outputs/tableau/.

Run from the repo root:   python src/export_tableau.py
Needs: duckdb, pandas, numpy, xgboost, requests, geopandas
"""
import glob, io, os, sys, zipfile
import duckdb
import numpy as np
import pandas as pd
import requests

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from zones import CBD_ZONES, BOUNDARY_ZONES

OUT = "outputs/tableau"
os.makedirs(OUT, exist_ok=True)
SUMMARY = "data/summary/*.parquet"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
CLUSTER_NAMES = {1: "Toll zone", 2: "Steady inner boroughs",
                 0: "Growing outer boroughs", 3: "Fast-growing periphery"}
GROUP_NAMES = {"inside": "Inside the zone", "entering": "Entering the zone",
               "exiting": "Exiting the zone", "outside": "Outside the zone",
               "boundary": "Boundary (60th St)"}
con = duckdb.connect()


def save(df, name):
    path = f"{OUT}/{name}"
    df.to_csv(path, index=False)
    print(f"  {name:32s} {len(df):>7,} rows")


lookup = pd.read_csv("data/taxi_zone_lookup.csv")
uptown = lookup[(lookup["Borough"] == "Manhattan")
                & (~lookup["LocationID"].isin(CBD_ZONES + BOUNDARY_ZONES))
                & (~lookup["LocationID"].isin([103, 104, 105]))]
uptown_ids = ",".join(map(str, uptown["LocationID"]))
print("Building Tableau extracts...")

# 1. Monthly toll revenue and trips, by company ------------------------------------------
monthly = con.sql(f"""
    SELECT date_trunc('month', trip_date) AS month, company,
           SUM(trips) AS trips, SUM(sum_cbd_fee) AS toll_revenue, SUM(tolled_trips) AS tolled_trips
    FROM '{SUMMARY}'
    WHERE company IN ('Uber', 'Lyft', 'Yellow')
    GROUP BY ALL ORDER BY month, company
""").df()
save(monthly, "monthly_toll_revenue.csv")

# 2. Weekly Uber trips by trip group ------------------------------------------------------
weekly_groups = con.sql(f"""
    SELECT date_trunc('week', trip_date) AS week, trip_group, SUM(trips) AS trips
    FROM '{SUMMARY}'
    WHERE company = 'Uber' AND trip_date >= '2023-01-02' AND trip_date < '2026-07-27'
    GROUP BY ALL ORDER BY week
""").df()
weekly_groups["trip_group"] = weekly_groups["trip_group"].map(GROUP_NAMES)
save(weekly_groups, "weekly_trips_by_group.csv")

# 3. Weekly actual vs ML "no toll" forecast (same model as notebook 03) --------------------
from xgboost import XGBRegressor
from pandas.tseries.holiday import USFederalHolidayCalendar

raw = con.sql(f"""
    SELECT trip_date, CASE WHEN trip_group = 'outside' THEN 'control' ELSE trip_group END AS grp,
           SUM(trips) AS trips
    FROM '{SUMMARY}'
    WHERE company = 'Uber'
      AND (trip_group IN ('inside', 'entering', 'exiting')
           OR (trip_group = 'outside' AND pu_zone IN ({uptown_ids})))
    GROUP BY ALL
""").df()
raw["trip_date"] = pd.to_datetime(raw["trip_date"])
wide = raw.pivot(index="trip_date", columns="grp", values="trips").sort_index()
hol = USFederalHolidayCalendar().holidays("2023-01-01", "2026-12-31")
GROUPS = ["inside", "entering", "exiting"]
df = wide[GROUPS].stack().rename("trips").reset_index()
df.columns = ["trip_date", "grp", "trips"]
df = df.merge(wide["control"].rename("control_trips"), left_on="trip_date", right_index=True)
df["dow"] = df["trip_date"].dt.dayofweek
df["month"] = df["trip_date"].dt.month
df["doy"] = df["trip_date"].dt.dayofyear
df["week"] = df["trip_date"].dt.isocalendar().week.astype(int)
df["is_holiday"] = df["trip_date"].isin(hol).astype(int)
df["near_holiday"] = df["trip_date"].apply(lambda d: int(any(abs((d - h).days) <= 2 for h in hol)))
df["holiday_season"] = ((df["month"] == 12) & (df["trip_date"].dt.day >= 20) |
                        (df["month"] == 1) & (df["trip_date"].dt.day <= 3)).astype(int)
df["t"] = (df["trip_date"] - pd.Timestamp("2023-01-01")).dt.days
df["log_control"] = np.log(df["control_trips"])
df["grp_code"] = df["grp"].map({g: i for i, g in enumerate(GROUPS)})
FEATURES = ["grp_code", "dow", "month", "doy", "week", "is_holiday", "near_holiday",
            "holiday_season", "t", "log_control"]
model = XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=4,
                     subsample=0.8, colsample_bytree=0.8, random_state=42)
train = df[df["trip_date"] < "2025-01-01"]
model.fit(train[FEATURES], np.log(train["trips"]))
df["forecast"] = np.exp(model.predict(df[FEATURES]))
df.loc[df["trip_date"] < "2025-01-05", "forecast"] = np.nan      # forecast only after the toll
wk = df.groupby(pd.Grouper(key="trip_date", freq="W-SUN"))[["trips", "forecast"]].sum(min_count=1)
wk = wk.iloc[1:-1].reset_index().rename(columns={"trip_date": "week_ending"})
cf = wk.melt(id_vars="week_ending", var_name="series", value_name="weekly_trips").dropna()
cf["series"] = cf["series"].map({"trips": "Actual", "forecast": "Forecast without toll"})
save(cf, "weekly_actual_vs_forecast.csv")

# 4. Effect estimates (evidence panel) ------------------------------------------------------
kpi = pd.read_csv("outputs/did_kpis_uptown_control_uber.csv")
kpi = kpi[kpi["group"] == "entering"].copy()
kpi["outcome"] = kpi["outcome"].map({"trips": "Trips", "avg_cost": "Rider cost per trip",
                                     "mph": "Traffic speed", "pay_hr": "Driver pay per hour"})
kpi["method"] = kpi["spec"].map({"plain": "Difference-in-differences",
                                 "with trends": "DiD + pre-existing trend"})
kpi["significant"] = np.where(kpi["p_value"] < 0.05, "Significant (p < 0.05)", "Not significant")
effects = kpi[["outcome", "method", "effect_pct", "ci_low", "ci_high", "p_value", "significant"]]
ml = pd.read_csv("outputs/ml_counterfactual_uber.csv")
pl = pd.read_csv("outputs/ml_placebo_uber.csv")
ml_e = ml[(ml["year"] == 2025) & (ml["grp"] == "entering")]["effect_pct"].iloc[0]
pl_e = pl[pl["grp"] == "entering"]["placebo_effect_pct"].iloc[0]
effects = pd.concat([effects, pd.DataFrame([{
    "outcome": "Trips", "method": "ML forecast (placebo-adjusted)",
    "effect_pct": ml_e - pl_e, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan,
    "significant": "Point estimate"}])], ignore_index=True)
save(effects.round(2), "effects_entering_trips.csv")

# 5. Hourly demand profile: toll zone vs rest of city ----------------------------------------
cbd = ",".join(map(str, CBD_ZONES))
hourly = con.sql(f"""
    SELECT hour,
           CASE WHEN pu_zone IN ({cbd}) THEN 'Toll zone' ELSE 'Rest of city' END AS area,
           CASE WHEN dayofweek(trip_date) IN (0, 6) THEN 'Weekend' ELSE 'Weekday' END AS day_type,
           SUM(trips) AS trips
    FROM '{SUMMARY}'
    WHERE company = 'Uber' AND trip_date BETWEEN '2025-02-01' AND '2025-12-31'
    GROUP BY ALL
""").df()
hourly["share_of_day_pct"] = hourly["trips"] / hourly.groupby(["area", "day_type"])["trips"].transform("sum") * 100
save(hourly.sort_values(["area", "day_type", "hour"]).round(3), "hourly_demand_profile.csv")

# 6. Cluster x time block: waits, margin, driver pay, dollar opportunity --------------------
zc = pd.read_csv("outputs/zone_clusters_uber.csv")
ids = ",".join(map(str, zc["LocationID"]))
blocks = con.sql(f"""
    SELECT pu_zone AS LocationID,
           CASE WHEN trip_date < DATE '2025-01-01' THEN '2024' ELSE '2025' END AS yr,
           CASE WHEN hour < 6 THEN '1 Night (12-6am)'
                WHEN hour < 10 THEN '2 Morning (6-10am)'
                WHEN hour < 16 THEN '3 Midday (10am-4pm)'
                WHEN hour < 20 THEN '4 Evening (4-8pm)'
                ELSE '5 Late (8pm-12am)' END AS time_block,
           SUM(trips) AS trips, SUM(sum_base_fare) AS base_fare, SUM(sum_driver_pay) AS driver_pay,
           SUM(sum_trip_seconds) AS secs, SUM(sum_wait_seconds) AS wait_s, SUM(wait_trips) AS wait_n
    FROM '{SUMMARY}'
    WHERE company = 'Uber' AND pu_zone IN ({ids})
      AND ((trip_date BETWEEN '2024-02-01' AND '2024-12-31')
        OR (trip_date BETWEEN '2025-02-01' AND '2025-12-31'))
    GROUP BY ALL
""").df().merge(zc[["LocationID", "cluster"]], on="LocationID")
cb = blocks.groupby(["cluster", "time_block", "yr"])[["trips", "base_fare", "driver_pay",
                                                        "secs", "wait_s", "wait_n"]].sum().unstack("yr")
o = pd.DataFrame(index=cb.index)
o["trips_2025"] = cb[("trips", "2025")]
o["trips_change_pct"] = (cb[("trips", "2025")] / cb[("trips", "2024")] - 1) * 100
o["wait_2025_min"] = cb[("wait_s", "2025")] / cb[("wait_n", "2025")] / 60
o["wait_change_min"] = o["wait_2025_min"] - cb[("wait_s", "2024")] / cb[("wait_n", "2024")] / 60
o["margin_per_trip"] = (cb[("base_fare", "2025")] - cb[("driver_pay", "2025")]) / cb[("trips", "2025")]
o["driver_pay_per_hr"] = cb[("driver_pay", "2025")] / (cb[("secs", "2025")] / 3600)
for label, rate in [("low", 0.02), ("mid", 0.05), ("high", 0.08)]:
    at_risk = o["trips_2025"] * o["wait_change_min"].clip(lower=0) * rate * 12 / 11
    o[f"opportunity_usd_{label}"] = at_risk * o["margin_per_trip"]
o = o.reset_index()
o.insert(1, "cluster_name", o["cluster"].map(CLUSTER_NAMES))
save(o.round(3), "cluster_time_opportunity.csv")

# 7. Headline KPI tiles --------------------------------------------------------------------
eff = effects.set_index(["outcome", "method"])["effect_pct"]
tiles = pd.DataFrame([
    {"order": 1, "kpi": "Toll paid on rides", "value": monthly["toll_revenue"].sum(),
     "format": "usd", "note": "Uber, Lyft & yellow taxis, Jan 2025 onward"},
    {"order": 2, "kpi": "Faster traffic", "value": eff[("Traffic speed", "Difference-in-differences")],
     "format": "pct", "note": "Uber trips entering the zone vs uptown Manhattan"},
    {"order": 3, "kpi": "Driver pay per hour", "value": eff[("Driver pay per hour", "Difference-in-differences")],
     "format": "pct", "note": "Uber trips entering the zone"},
    {"order": 4, "kpi": "Trips into the zone", "value": eff[("Trips", "ML forecast (placebo-adjusted)")],
     "format": "pct", "note": "Range across 3 methods: about -3% to -7%"},
    {"order": 5, "kpi": "Driver repositioning upside", "value": o["opportunity_usd_mid"].sum(),
     "format": "usd", "note": "Annual platform margin (range: low to high assumption)"},
])
save(tiles.round(2), "kpi_tiles.csv")

# 8. Zone map: TLC taxi zone shapes + cluster results, as GeoJSON (WGS84) --------------------
import geopandas as gpd
shp_dir = "data/taxi_zones_shp"
if not glob.glob(f"{shp_dir}/**/*.shp", recursive=True):
    r = requests.get("https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip",
                     headers=HEADERS, timeout=120)
    r.raise_for_status()
    zipfile.ZipFile(io.BytesIO(r.content)).extractall(shp_dir)
shp = glob.glob(f"{shp_dir}/**/*.shp", recursive=True)[0]
gdf = gpd.read_file(shp).to_crs(4326)
gdf["geometry"] = gdf["geometry"].simplify(0.0001, preserve_topology=True)
gdf = gdf[["LocationID", "zone", "borough", "geometry"]]
m = zc.drop(columns=["Zone", "Borough"], errors="ignore").copy()
m["cluster_name"] = m["cluster"].map(CLUSTER_NAMES)
gdf = gdf.merge(m, on="LocationID", how="left")
gdf["cluster_name"] = gdf["cluster_name"].fillna("Too few trips")
gdf["in_toll_zone"] = np.where(gdf["LocationID"].isin(CBD_ZONES), "Toll zone", "Outside")
gdf.to_file(f"{OUT}/nyc_zones_map.geojson", driver="GeoJSON")
print(f"  {'nyc_zones_map.geojson':32s} {len(gdf):>7,} shapes")

print("\nDone. Files are in", OUT)
