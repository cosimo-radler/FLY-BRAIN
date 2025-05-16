# Archived Source Files

This directory contains archived source files that are no longer used in the main workflow.

## Why were these files archived?

These files were created during the development phase of our integration with the neuPrint API. They used direct HTTP requests to the API instead of the official neuprint-python client library.

They have been replaced by the `neuprint_client.py` implementation which provides a more efficient and standardized way to interact with the neuPrint database.

## Should I use these files?

No. These files are kept for reference only. For all data retrieval, please use:

```python
from src.neuprint_client import NeuPrintInterface
```

For more information, please see the main [data_retrieval.md](../../data_retrieval.md) documentation. 