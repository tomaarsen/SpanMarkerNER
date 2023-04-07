MODEL_CARD_TEMPLATE = """
---
license: apache-2.0
library_name: span_marker
tags:
- span_marker
- token-classification
- ner
- named-entity-recognition
pipeline_tag: token-classification
---

# SpanMarker for Named Entity Recognition

This is a [SpanMarker](https://github.com/tomaarsen/SpanMarkerNER) model that can be used for Named Entity Recognition. \
In particular, this SpanMarker model uses {encoder_name_or_path} as the underlying encoder.

## Usage

To use this model for inference, first install the `span_marker` library:

```bash
pip install span_marker
```

You can then run inference as follows:

```python
from span_marker import SpanMarkerModel

# Download from Hub and run inference
model = SpanMarkerModel.from_pretrained("span_marker_model_name")
# Run inference
preds = model.predict("Amelia Earhart flew her single engine Lockheed Vega 5B across the Atlantic to Paris.")
```

See the [SpanMarker](https://github.com/tomaarsen/SpanMarkerNER) repository for documentation and additional information on this model framework.
"""
