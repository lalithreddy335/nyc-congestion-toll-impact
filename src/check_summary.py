import duckdb, sys

month = sys.argv[1] if len(sys.argv) > 1 else "2025-01"

print(duckdb.sql(f"""
    SELECT company, trip_group,
           SUM(trips) AS trips,
           ROUND(SUM(sum_cbd_fee)) AS toll,
           ROUND(SUM(sum_rider_cost) / SUM(trips), 2) AS avg_cost,
           ROUND(SUM(sum_miles) / (SUM(sum_trip_seconds) / 3600), 1) AS mph
    FROM 'data/summary/*{month}.parquet'
    GROUP BY ALL
    ORDER BY company, trips DESC
"""))