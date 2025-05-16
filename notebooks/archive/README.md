# Archived Scripts

This directory contains archived scripts that are no longer used in the main workflow.

## Why were these scripts archived?

These scripts were created during the development and testing phase of our neuPrint API integration. They have been replaced by the unified `neuprint_client.py` implementation which serves as the canonical way to fetch data from the neuPrint database.

## Should I use these scripts?

No. These scripts are kept for reference only. For all data retrieval, please use:

```python
from src.neuprint_client import NeuPrintInterface
```

And execute the canonical data fetching script:

```bash
python notebooks/fetch_all_regions.py
```

For more information, please see the main [data_retrieval.md](../../data_retrieval.md) documentation. 