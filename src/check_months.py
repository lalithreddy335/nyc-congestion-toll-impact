import duckdb, glob

files = glob.glob("data/summary/*.parquet")
print("Summary files:", len(files), "(expect 86)")

print(duckdb.sql("""
    SELECT strftime(trip_date, '%Y-%m') AS month,
           ROUND(SUM(trips) FILTER (WHERE company IN ('Uber','Lyft')) / 1e6, 2) AS rideshare_M,
           ROUND(SUM(trips) FILTER (WHERE company = 'Yellow') / 1e6, 2) AS yellow_M,
           ROUND(SUM(sum_cbd_fee) / 1e6, 2) AS toll_M
    FROM 'data/summary/*.parquet'
    GROUP BY 1
    ORDER BY 1
""").show(max_rows=60))