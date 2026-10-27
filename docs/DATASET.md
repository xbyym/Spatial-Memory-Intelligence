# Training data

| Backbone | Training rows | Evaluation rows |
| --- | ---: | ---: |
| HY1.5 | 63,916 | 191 |
| Wan2.2 | 30,936 | 489 |

The data covers spatial clustering, memory sparsification, action-aware retrieval, and reliability filtering. Image paths are relative to the data root. Identical image contents are stored once; repeated training examples are retained in the annotations.

Download and extract the data:

```bash
python scripts/download_assets.py --component data --extract
```

The prepared data contains 67,827 unique images. Downloaded files are verified before extraction. Training and evaluation annotations are separate. Training commands are provided in the main README.
