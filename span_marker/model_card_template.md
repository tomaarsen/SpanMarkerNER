---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
{{ card_data }}
---

# {{ model_name | default("SpanMarker for Named Entity Recognition", true) }}

This is a [SpanMarker](https://github.com/tomaarsen/SpanMarkerNER) model{% if dataset_id %} trained on the [{{ dataset_name if dataset_name else dataset_id }}](https://huggingface.co/datasets/{{ dataset_id }}) dataset{% endif %} that can be used for {{ task_name | default("Named Entity Recognition", true) }}.{% if encoder_id %} This SpanMarker model uses [{{ encoder_name if encoder_name else encoder_id }}](https://huggingface.co/{{ encoder_id }}) as the underlying encoder.{% endif %}

## Model Details

### Model Description
- **Model Type:** SpanMarker
{% if encoder_id -%}
    - **Encoder:** [{{ encoder_name if encoder_name else encoder_id }}](https://huggingface.co/{{ encoder_id }})
{%- else -%}
    <!-- - **Encoder:** [Unknown](https://huggingface.co/unknown) -->
{%- endif %}
- **Maximum Sequence Length:** {{ model_max_length }} tokens
- **Maximum Entity Length:** {{ entity_max_length }} words
{% if dataset_id -%}
    - **Training Dataset:** [{{ dataset_name if dataset_name else dataset_id }}](https://huggingface.co/datasets/{{ dataset_id }})
{%- else -%}
    <!-- - **Training Dataset:** [Unknown](https://huggingface.co/datasets/unknown) -->
{%- endif %}
{% if language -%}
    - **Language{{"s" if language is not string and language | length > 1 else ""}}:**
    {%- if language is string %} {{ language }}
    {%- else %} {% for lang in language -%}
            {{ lang }}{{ ", " if not loop.last else "" }}
        {%- endfor %}
    {%- endif %}
{%- else -%}
    <!-- - **Language:** Unknown -->
{%- endif %}
{% if license -%}
    - **License:** {{ license }}
{%- else -%}
    <!-- - **License:** Unknown -->
{%- endif %}

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

### Direct Use for Inference

```python
from span_marker import SpanMarkerModel

# Download from the {{ hf_emoji }} Hub
model = SpanMarkerModel.from_pretrained("{{ model_id | default('span_marker_model_id', true) }}")
# Run inference
entities = model.predict("{{ predict_example | replace('"', '\\"') | default("Amelia Earhart flew her single engine Lockheed Vega 5B across the Atlantic to Paris.", true)}}")
```

### Downstream Use
You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

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
</details>

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details
{% if train_set_metrics %}
### Training Set Metrics
{{ train_set_metrics }}{% endif %}{% if hyperparameters %}
### Training Hyperparameters
{% for name, value in hyperparameters.items() %}- {{ name }}: {{ value }}
{% endfor %}{% endif %}{% if eval_lines %}
### Training Results
{{ eval_lines }}{% endif %}{% if co2_eq_emissions %}
### Environmental Impact
Carbon emissions were measured using [CodeCarbon](https://github.com/mlco2/codecarbon).
- **Carbon Emitted**: {{ "%.3f"|format(co2_eq_emissions["emissions"] / 1000) }} kg of CO2
- **Hours Used**: {{ co2_eq_emissions["hours_used"] }} hours

### Training Hardware
- **On Cloud**: {{ "Yes" if co2_eq_emissions["on_cloud"] else "No" }}
- **GPU Model**: {{ co2_eq_emissions["hardware_used"] or "No GPU used" }}
- **CPU Model**: {{ co2_eq_emissions["cpu_model"] }}
- **RAM Size**: {{ "%.2f"|format(co2_eq_emissions["ram_total_size"]) }} GB
{% endif %}
### Framework Versions
- Python: {{ version["python"] }}
- SpanMarker: {{ version["span_marker"] }}
- Transformers: {{ version["transformers"] }}
- PyTorch: {{ version["torch"] }}
- Datasets: {{ version["datasets"] }}
- Tokenizers: {{ version["tokenizers"] }}

## Citation

### BibTeX
```
@software{Aarsen_SpanMarker,
    author = {Aarsen, Tom},
    license = {Apache-2.0},
    title = {{"{{SpanMarker for Named Entity Recognition}}"}},
    url = {https://github.com/tomaarsen/SpanMarkerNER}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->