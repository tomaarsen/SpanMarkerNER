import os
from pathlib import Path
from typing import Union

import jinja2
from huggingface_hub import model_info
from huggingface_hub.utils import RepositoryNotFoundError

from span_marker.configuration import SpanMarkerConfig

MODEL_CARD_TEMPLATE = """
---
license: apache-2.0
library_name: span-marker
tags:
- span-marker
- token-classification
- ner
- named-entity-recognition
pipeline_tag: token-classification
---

# SpanMarker for Named Entity Recognition

This is a [SpanMarker](https://github.com/tomaarsen/SpanMarkerNER) model that can be used \
for Named Entity Recognition. {% if encoder_name_or_path %}In particular, this SpanMarker model uses \
{% if is_public_model %}\
[{{ encoder_name_or_path }}](https://huggingface.co/{{ encoder_name_or_path }})\
{% else %}\
"{{ encoder_name_or_path }}"\
{% endif %} as the underlying encoder. {% endif %}

## Usage

To use this model for inference, first install the `span_marker` library:

```bash
pip install span_marker
```

You can then run inference with this model like so:

```python
from span_marker import SpanMarkerModel

# Download from the 🤗 Hub
model = SpanMarkerModel.from_pretrained({% if model_name_or_path %}"{{ model_name_or_path }}"{% else %}"span_marker_model_name"{% endif %})
# Run inference
entities = model.predict("Amelia Earhart flew her single engine Lockheed Vega 5B across the Atlantic to Paris.")
```

See the [SpanMarker](https://github.com/tomaarsen/SpanMarkerNER) repository for documentation and additional information on this library.
"""


def is_public_model(encoder_name_or_path: str) -> bool:
    # Models with more than two 'sections' certainly are not public models
    if len(encoder_name_or_path.split("/")) > 2:
        return False

    try:
        model_info(encoder_name_or_path)
        return True
    except RepositoryNotFoundError:
        return False


def generate_model_card(save_directory: Union[str, os.PathLike], config: SpanMarkerConfig) -> str:
    template = jinja2.Environment().from_string(MODEL_CARD_TEMPLATE)
    save_directory = Path(save_directory)
    context = {}

    context["model_name_or_path"] = "span_marker_model_name"

    if "_name_or_path" in config.encoder:
        context["encoder_name_or_path"] = config.encoder["_name_or_path"]
        context["is_public_model"] = is_public_model(context["encoder_name_or_path"])

    return template.render(context)
