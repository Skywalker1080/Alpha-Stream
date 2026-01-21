from dask.dot import name
from datetime import timedelta
from feast import FeatureStore, FeatureView, Field, FileSource, ValueType, Entity, Project
from feast.types import Float32

# Defining Entity (primary key)
ticker = Entity(name="ticker", join_keys=["ticker"], value_type=ValueType.STRING)

# Defining Project
project = Project(name="mlops_pipeline", description="feature store for alpha server")

# Defining File source
file_source = FileSource(
    name="crypto_source",
    path="data/crypto_data.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# Defining Feature View (Data Schema)
feature_view = FeatureView(
    name="crypto_features",
    entities=[ticker],
    ttl=timedelta(days=36500),
    schema=[
        Field(name="Open", dtype=Float32),
        Field(name="High", dtype=Float32),
        Field(name="Low", dtype=Float32),
        Field(name="Close", dtype=Float32),
        Field(name="Volume", dtype=Float32),
        Field(name="RSI", dtype=Float32),
        Field(name="MACD", dtype=Float32),
    ],
    online=True,
    offline=True,
    source=file_source,
)