#!/usr/bin/env python
"""Run the water balance model on a test area."""

from datetime import date

from WaterBalanceModel.query import WaterBalanceQuery
from WaterBalanceModel.run_model import run_pipeline

# Create query for a small test area in NSW, Australia
# This is near Canowindra - a hilly agricultural area
query = WaterBalanceQuery.from_lat_lon(
    lat=-33.51,
    lon=148.37,
    buffer_km=1.0,  # 2km x 2km area
    start=date(2023, 1, 1),
    end=date(2023, 3, 31),  # 3 months
    stub='test_canowindra',
)

print(f'Query: {query.stub}')
print(f'  bbox: {query.bbox}')
print(f'  period: {query.start} to {query.end}')
print(f'  output: {query.soil_moisture_path}')
print()

# Run the pipeline
result = run_pipeline(query)

print()
print('Result:')
print(result)
