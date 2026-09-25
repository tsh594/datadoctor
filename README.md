# Data Doctor

An automated data cleaning and ML pipeline preparation library.

## Installation

```bash
pip install datadoctor
```

## Quick Start

```python
import pandas as pd
from datadoctor import Doctor

df = pd.read_csv("messy_data.csv")
doc = Doctor(df)
doc.diagnose()
cleaned_df = doc.treat()
doc.export_model(target_column="Attrition")
```

## License

Apache 2.0