import duckdb, os, sys, time
import requests
from datetime import date
from zones import CBD_ZONES, BOUNDARY_ZONES

BASE = "https://d37ci6vzurychx.cloudfront.net/trip-data"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "summary")
RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")
os.makedirs(OUT, exist_ok=True)
os.makedirs(RAW, exist_ok=True)

con = duckdb.connect()
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def download(url, path):
    """Stream a file to disk. Returns False if it doesn't exist (403/404)."""
    with requests.get(url, headers=HEADERS, stream=True, timeout=60) as r:
        if r.status_code in (403, 404):
            return False
        r.raise_for_status()
        tmp = path + ".part"
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)
    os.replace(tmp, path)
    return True

cbd = ",".join(map(str, CBD_ZONES))
bnd = ",".join(map(str, BOUNDARY_ZONES))

TRIP_GROUP = f"""
CASE
  WHEN PULocationID IN ({bnd}) OR DOLocationID IN ({bnd}) THEN 'boundary'
  WHEN PULocationID IN ({cbd}) AND DOLocationID IN ({cbd}) THEN 'inside'
  WHEN DOLocationID IN ({cbd}) THEN 'entering'
  WHEN PULocationID IN ({cbd}) THEN 'exiting'
  ELSE 'outside'
END"""


def cbd_expr(url):
    cols = {r[0].lower() for r in con.sql(f"DESCRIBE SELECT * FROM '{url}'").fetchall()}
    return "COALESCE(cbd_congestion_fee, 0)" if "cbd_congestion_fee" in cols else "0"


def hvfhv_sql(url, start, end, out):
    fee = cbd_expr(url)
    return f"""
    COPY (
      SELECT
        CAST(pickup_datetime AS DATE) AS trip_date,
        HOUR(pickup_datetime) AS hour,
        CASE hvfhs_license_num WHEN 'HV0003' THEN 'Uber'
                               WHEN 'HV0005' THEN 'Lyft' ELSE 'Other' END AS company,
        PULocationID AS pu_zone,
        {TRIP_GROUP} AS trip_group,
        COUNT(*) AS trips,
        SUM(trip_miles) AS sum_miles,
        SUM(trip_time) AS sum_trip_seconds,
        SUM(base_passenger_fare + tolls + bcf + sales_tax + congestion_surcharge
            + COALESCE(airport_fee, 0) + {fee}) AS sum_rider_cost,
        SUM(base_passenger_fare) AS sum_base_fare,
        SUM(driver_pay) AS sum_driver_pay,
        SUM(tips) AS sum_tips,
        SUM({fee}) AS sum_cbd_fee,
        SUM(CASE WHEN {fee} > 0 THEN 1 ELSE 0 END) AS tolled_trips,
        SUM(date_diff('second', request_datetime, pickup_datetime))
            FILTER (WHERE pickup_datetime >= request_datetime
                    AND date_diff('second', request_datetime, pickup_datetime) < 3600)
            AS sum_wait_seconds,
        COUNT(*) FILTER (WHERE pickup_datetime >= request_datetime
                    AND date_diff('second', request_datetime, pickup_datetime) < 3600)
            AS wait_trips
      FROM '{url}'
      WHERE pickup_datetime >= DATE '{start}' AND pickup_datetime < DATE '{end}'
        AND trip_miles > 0 AND trip_time > 0 AND trip_time < 14400
        AND base_passenger_fare > 0
        AND trip_miles < 100 AND trip_miles / (trip_time / 3600.0) < 80
        AND PULocationID < 264 AND DOLocationID < 264
      GROUP BY ALL
    ) TO '{out}' (FORMAT parquet)"""


def yellow_sql(url, start, end, out):
    fee = cbd_expr(url)
    secs = "date_diff('second', tpep_pickup_datetime, tpep_dropoff_datetime)"
    return f"""
    COPY (
      SELECT
        CAST(tpep_pickup_datetime AS DATE) AS trip_date,
        HOUR(tpep_pickup_datetime) AS hour,
        'Yellow' AS company,
        PULocationID AS pu_zone,
        {TRIP_GROUP} AS trip_group,
        COUNT(*) AS trips,
        SUM(trip_distance) AS sum_miles,
        SUM({secs}) AS sum_trip_seconds,
        SUM(total_amount - tip_amount) AS sum_rider_cost,
        SUM(fare_amount) AS sum_base_fare,
        NULL::DOUBLE AS sum_driver_pay,
        SUM(tip_amount) AS sum_tips,
        SUM({fee}) AS sum_cbd_fee,
        SUM(CASE WHEN {fee} > 0 THEN 1 ELSE 0 END) AS tolled_trips,
        NULL::DOUBLE AS sum_wait_seconds,
        NULL::BIGINT AS wait_trips
      FROM '{url}'
      WHERE tpep_pickup_datetime >= DATE '{start}' AND tpep_pickup_datetime < DATE '{end}'
        AND trip_distance > 0 AND fare_amount > 0
        AND {secs} > 0 AND {secs} < 14400
        AND trip_distance < 100 AND trip_distance / ({secs} / 3600.0) < 80
        AND PULocationID < 264 AND DOLocationID < 264
      GROUP BY ALL
    ) TO '{out}' (FORMAT parquet)"""


def month_list():
    if len(sys.argv) > 1:                      # test mode: python build_summary.py 2025-01
        y, m = map(int, sys.argv[1].split("-"))
        return [(y, m)]
    y, m, today, months = 2023, 1, date.today(), []
    while (y, m) <= (today.year, today.month):
        months.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return months


DATASETS = {"fhvhv": hvfhv_sql, "yellow": yellow_sql}

for y, m in month_list():
    ym = f"{y}-{m:02d}"
    start = f"{ym}-01"
    end = f"{y + 1}-01-01" if m == 12 else f"{y}-{m + 1:02d}-01"
    for name, build in DATASETS.items():
        out = os.path.join(OUT, f"{name}_{ym}.parquet").replace("\\", "/")
        if os.path.exists(out):
            print(f"{name} {ym}: already done, skipping")
            continue
        url = f"{BASE}/{name}_tripdata_{ym}.parquet"
        raw = os.path.join(RAW, f"{name}_{ym}.parquet").replace("\\", "/")
        for attempt in range(1, 4):            # up to 3 tries
            t = time.time()
            try:
                if not os.path.exists(raw):
                    if not download(url, raw):
                        print(f"{name} {ym}: not published yet, skipping")
                        break
                con.sql(build(raw, start, end, out))
                os.remove(raw)                 # delete the big raw file, keep the summary
                print(f"{name} {ym}: done in {time.time() - t:.0f}s")
                break
            except Exception as e:
                msg = str(e).splitlines()[0][:200]
                print(f"{name} {ym}: FAILED (try {attempt}) -> {msg}")
                for p in (raw, raw + ".part"):
                    if os.path.exists(p):
                        os.remove(p)
                time.sleep(15 * attempt)       # wait longer each retry
