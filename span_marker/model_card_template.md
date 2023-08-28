---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
{{ card_data }}
---

# {{ model_name | default("SpanMarker for Named Entity Recognition", true) }}

This is a [SpanMarker](https://github.com/tomaarsen/SpanMarkerNER) model{% if dataset_id %} trained on the [{{ dataset_name if dataset_name else dataset_id }}](https://huggingface.co/datasets/{{ dataset_id }}) dataset{% endif %} that can be used for {{ task_name | default("Named Entity Recognition", true) }}.{% if encoder_id %} This SpanMarker model uses [{{ encoder_name if encoder_name else encoder_id }}](https://huggingface.co/models/{{ encoder_id }}) as the underlying encoder.{% endif %}

## Model Details

### Model Description

- **Model Type:** SpanMarker
{% if encoder_id %}- **Encoder:** [{{ encoder_name if encoder_name else encoder_id }}](https://huggingface.co/models/{{ encoder_id }}){% endif %}
- **Maximum Sequence Length:** {{ model_max_length }} tokens
- **Maximum Entity Length:** {{ entity_max_length }} words
{% if dataset_id %}- **Training Dataset:** [{{ dataset_name if dataset_name else dataset_id }}](https://huggingface.co/datasets/{{ dataset_id }}){% endif %}
{% if language -%}
- **Language{{"s" if language is not string and language | length > 1 else ""}}:**
{%- if language is string %} {{ language }}
{%- else %}{% for lang in language %} {{ lang }}{{ "," if not loop.last else "" }}{% endfor %}
{%- endif %}
{%- endif %}
{% if license %}- **License:** {{ license }}{% endif %}

### Model Sources

- **Repository:** [SpanMarker on GitHub](https://github.com/tomaarsen/SpanMarkerNER)
- **Thesis:** [SpanMarker For Named Entity Recognition](https://raw.githubusercontent.com/tomaarsen/SpanMarkerNER/main/thesis.pdf)
{% if label_examples %}
### Model Labels
{{ label_examples }}{% endif -%}
{% if metrics_table %}
## Evaluation

### Metrics
{{ metrics_table }}{% endif %}
## Uses

### Direct Use

```python
from span_marker import SpanMarkerModel

# Download from the {{ hf_emoji }} Hub
model = SpanMarkerModel.from_pretrained("{{ model_id | default('span_marker_model_id', true) }}")
# Run inference
entities = model.predict("{{ predict_example | default("Amelia Earhart flew her single engine Lockheed Vega 5B across the Atlantic to Paris.", true)}}")
```

### Downstream Use
You can finetune this model on your own dataset.

```python
from span_marker import SpanMarkerModel, Trainer

# Download from the {{ hf_emoji }} Hub
model = SpanMarkerModel.from_pretrained("{{ model_id | default('span_marker_model_id', true) }}")

# Specify a Dataset with "tokens" and "ner_tag" columns
dataset = load_dataset("conll2003") # For example CoNLL2003

# Initialize a Trainer using the pretrained model & dataset
trainer = Trainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)
trainer.train()
trainer.save_model("{{ model_id | default('span_marker_model_id', true) }}-finetuned")
```
{% if tokenizer_warning %}
### {{ warn_emoji }} Tokenizer Warning
The [{{ encoder_name if encoder_name else encoder_id }}](https://huggingface.co/models/{{ encoder_id }}) tokenizer distinguishes between punctuation directly attached to a word and punctuation separated from a word by a space. For example, `Paris.` and `Paris .` are tokenized into different tokens. During training, this model is only exposed to the latter style, i.e. all words are separated by a space. Consequently, the model may perform worse when the inference text is in the former style.

In short, it is recommended to preprocess your inference text such that all words and punctuation are separated by a space. Some potential approaches to convert regular text into this format are NLTK [`word_tokenize`](https://www.nltk.org/api/nltk.tokenize.word_tokenize.html) or spaCy [`Doc`](https://spacy.io/api/doc#iter) and join the resulting words with a space.
{% endif %}
## Training Details
{% if train_set_metrics %}
### Training Set metrics
{{ train_set_metrics }}{% endif %}{% if hyperparameters %}
### Training Hyperparameters

{% for name, value in hyperparameters.items() %}- {{ name }}: {{ value }}
{% endfor %}{% endif %}{% if eval_lines %}
### Training Results

{{ eval_lines }}{% endif %}
### Framework Versions

- SpanMarker: {{ version["span_marker"] }}
- Transformers : {{ version["transformers"] }}
- PyTorch: {{ version["torch"] }}
- Datasets: {{ version["datasets"] }}
- Tokenizers: {{ version["tokenizers"] }}
