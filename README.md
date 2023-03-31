# SpanMarker for Named Entity Recognition

SpanMarker is a framework for training powerful Named Entity Recognition models using familiar encoders such as BERT, RoBERTa and DeBERTa.
Tightly implemented on top of the [🤗 Transformers](https://github.com/huggingface/transformers/) library, SpanMarker can take advantage of its valuable functionality.
<!-- like performance dashboard integration, automatic mixed precision, 8-bit inference-->

Based on the [PL-Marker](https://arxiv.org/pdf/2109.06067.pdf) paper, SpanMarker breaks the mold through its accessibility and ease of use. Crucially, SpanMarker works out of the box with many common encoders such as `bert-base-cased` and `roberta-large`, and automatically works with datasets using the `IOB`, `IOB2`, `BIOES`, `BILOU` or no label annotation scheme.

## Installation
You may install the `span_marker` Python module via `pip` like so:
```
pip install span_marker
```

## Quick Start
Please have a look at our [Getting Started](examples/getting_started.ipynb) jupyter notebook for details on how SpanMarker is commonly used. That notebook explains the following snippet in more detail.

```python
from datasets import load_dataset
from span_marker import SpanMarkerModel, Trainer
from transformers import TrainingArguments

dataset = load_dataset("DFKI-SLT/few-nerd", "supervised")
labels = dataset["train"].features["ner_tags"].feature.names

model_name = "bert-base-cased"
model = SpanMarkerModel.from_pretrained(model_name, labels=labels)

args = TrainingArguments(
    output_dir="my_span_marker_model",
    learning_rate=5e-5,
    gradient_accumulation_steps=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    save_strategy="steps",
    eval_steps=200,
    logging_steps=50,
    fp16=True,
    warmup_ratio=0.1,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"].select(range(8000)),
    eval_dataset=dataset["validation"].select(range(2000)),
)

trainer.train()
trainer.save_model("my_span_marker_model/checkpoint-final")

metrics = trainer.evaluate()
print(metrics)
```

For this work is based on [PL-Marker](https://arxiv.org/pdf/2109.06067v5.pdf), you may expect similar results to its [Papers with Code Leaderboard](https://paperswithcode.com/paper/pack-together-entity-and-relation-extraction). Tests, documentation and further information on expected performance will come soon.

## Pretrained Models

* [`tomaarsen/span-marker-bert-base-fewnerd-fine-super`](https://huggingface.co/tomaarsen/span-marker-bert-base-fewnerd-fine-super) is a model that I have trained in just 4 hours on the finegrained, supervised [Few-NERD dataset](https://huggingface.co/datasets/DFKI-SLT/few-nerd). It reached a 0.7020 Test F1, competitive in the all-time [Few-NERD leaderboard](https://paperswithcode.com/sota/named-entity-recognition-on-few-nerd-sup).
See this [Weights and Biases report](https://api.wandb.ai/links/tomaarsen/dm21vbbm) for details. You can observe the model inferences in this [Argilla dataset](https://argilla-span-marker.hf.space/datasets/team/span-marker-bert-base-fewnerd-fine-super) (username: `argilla`, password: `1234`).

