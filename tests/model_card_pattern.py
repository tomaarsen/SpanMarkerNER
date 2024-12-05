import re

MODEL_CARD_PATTERN = re.compile(
    """\
---
language:
- en
license: apache-2\.0
tags:
- span-marker
- token-classification
- ner
- named-entity-recognition
- generated_from_span_marker_trainer
widget:
- text: .*
pipeline_tag: token-classification
library_name: span-marker
metrics:
- precision
- recall
- f1
co2_eq_emissions:
  emissions: [\d\.\-e]+
  source: codecarbon
  training_type: fine-tuning
  on_cloud: (false|true)
  cpu_model: .+
  ram_total_size: [\d\.]+
  hours_used: [\d\.]+
(  hardware_used: .+
)?datasets:
- conll2003
base_model: prajjwal1/bert-tiny
model-index:
- name: SpanMarker with prajjwal1/bert-tiny on CoNLL 2003
  results:
  - task:
      type: token-classification
      name: Named Entity Recognition
    dataset:
      name: CoNLL 2003
      type: conll2003
      split: eval
    metrics:
    - type: f1
      value: [\d\.]+
      name: F1
    - type: precision
      value: [\d\.]+
      name: Precision
    - type: recall
      value: [\d\.]+
      name: Recall
---

# SpanMarker with prajjwal1/bert-tiny on CoNLL 2003

This is a \[SpanMarker\]\(https://github.com/tomaarsen/SpanMarkerNER\) model trained on the \[CoNLL 2003\]\(https://huggingface.co/datasets/conll2003\) dataset that can be used for Named Entity Recognition. This SpanMarker model uses \[prajjwal1/bert-tiny\]\(https://huggingface.co/prajjwal1/bert-tiny\) as the underlying encoder.

## Model Details

### Model Description
- \*\*Model Type:\*\* SpanMarker
- \*\*Encoder:\*\* \[prajjwal1/bert-tiny\]\(https://huggingface.co/prajjwal1/bert-tiny\)
- \*\*Maximum Sequence Length:\*\* 512 tokens
- \*\*Maximum Entity Length:\*\* 8 words
- \*\*Training Dataset:\*\* \[CoNLL 2003\]\(https://huggingface.co/datasets/conll2003\)
- \*\*Language:\*\* en
- \*\*License:\*\* apache-2.0

### Model Sources

- \*\*Repository:\*\* \[SpanMarker on GitHub\]\(https://github.com/tomaarsen/SpanMarkerNER\)
- \*\*Thesis:\*\* \[SpanMarker For Named Entity Recognition\]\(https://raw.githubusercontent.com/tomaarsen/SpanMarkerNER/main/thesis.pdf\)

### Model Labels
\| Label        \| Examples                                    \|
\|:-------------\|:--------------------------------------------\|
\| art          \|                                             \|
\| building     \|                                             \|
\| event        \|                                             \|
\| location     \|                                             \|
\| organization \|                                             \|
\| other        \|                                             \|
\| person       \| [^\|]+ \|
\| product      \|                                             \|

## Uses

### Direct Use for Inference

```python
from span_marker import SpanMarkerModel

# Download from the [^H]+ Hub
model = SpanMarkerModel.from_pretrained\("tomaarsen/span-marker-test-model-card"\)
# Run inference
entities = model.predict\(".+"\)
```

### Downstream Use
You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

```python
from span_marker import SpanMarkerModel, Trainer

# Download from the [^H]+ Hub
model = SpanMarkerModel.from_pretrained\("tomaarsen/span-marker-test-model-card"\)

# Specify a Dataset with "tokens" and "ner_tag" columns
dataset = load_dataset\("conll2003"\) # For example CoNLL2003

# Initialize a Trainer using the pretrained model & dataset
trainer = Trainer\(
    model=model,
    train_dataset=dataset\["train"\],
    eval_dataset=dataset\["validation"\],
\)
trainer.train\(\)
trainer.save_model\("tomaarsen/span-marker-test-model-card-finetuned"\)
```
</details>

<!--
### Out-of-Scope Use

\*List how the model may foreseeably be misused and address what users ought not to do with the model\.\*
-->

<!--
## Bias, Risks and Limitations

\*What are the known or foreseeable issues stemming from this model\? You could also flag here known failure cases or weaknesses of the model\.\*
-->

<!--
### Recommendations

\*What are recommendations with respect to the foreseeable issues\? For example, filtering explicit content\.\*
-->

## Training Details

### Training Set Metrics
\| Training set          \| Min \| Median \| Max \|
\|:----------------------\|:----\|:-------\|:----\|
\| Sentence length       \| 4   \| 8.0    \| 12  \|
\| Entities per sentence \| 0   \| 1.5    \| 3   \|

### Training Hyperparameters
- learning_rate: 5e-05
- train_batch_size: 1
- eval_batch_size: 8
- seed: 42
- optimizer: Use adamw_torch with betas=\(0\.9,0\.999\) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- num_epochs: 1

### Training Results
\| Epoch \| Step \| Validation Loss \| Validation Precision \| Validation Recall \| Validation F1 \| Validation Accuracy \|
\|:-----:\|:----:\|:---------------:\|:--------------------:\|:-----------------:\|:-------------:\|:-------------------:\|
\| [\d\.]+ +\| [\d\.]+ +\| [\d\.]+ +\| [\d\.]+ +\| [\d\.]+ +\| [\d\.]+ +\| [\d\.]+ +\|
\| [\d\.]+ +\| [\d\.]+ +\| [\d\.]+ +\| [\d\.]+ +\| [\d\.]+ +\| [\d\.]+ +\| [\d\.]+ +\|

### Environmental Impact
Carbon emissions were measured using \[CodeCarbon\]\(https://github.com/mlco2/codecarbon\)\.
- \*\*Carbon Emitted\*\*: [\d\.]+ kg of CO2
- \*\*Hours Used\*\*: [\d\.]+ hours

### Training Hardware
- \*\*On Cloud\*\*: (Yes|No)
- \*\*GPU Model\*\*: [^\n]+
- \*\*CPU Model\*\*: [^\n]+
- \*\*RAM Size\*\*: [\d\.]+ GB

### Framework Versions
- Python: [^\n]+
- SpanMarker: [^\n]+
- Transformers: [^\n]+
- PyTorch: [^\n]+
- Datasets: [^\n]+
- Tokenizers: [^\n]+

## Citation

### BibTeX
```
@software{Aarsen_SpanMarker,
    author = {Aarsen, Tom},
    license = {Apache-2.0},
    title = {{SpanMarker for Named Entity Recognition}},
    url = {https://github.com/tomaarsen/SpanMarkerNER}
}
```

<!--
## Glossary

\*Clearly define terms in order to be accessible across audiences\.\*
-->

<!--
## Model Card Authors

\*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction\.\*
-->

<!--
## Model Card Contact

\*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors\.\*
-->""",
    flags=re.DOTALL,
)
