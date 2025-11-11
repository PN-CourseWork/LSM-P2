# Data Directory

This directory contains generated data files.

## Structure

Each example should create its own subdirectory:

```
data/
├── example_integration/
│   └── integration_results.parquet
└── your_example/
    └── your_data.parquet
```

## Note

Large data files should not be tracked. Instead create post-processed data files. 
