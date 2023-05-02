<h1 align="center">
SpanMarker for Named Entity Recognition
</h1>
<div align="center">

[🤗 Models](https://huggingface.co/models?other=span-marker) |
[🛠️ Getting Started In Google Colab](https://colab.research.google.com/github/tomaarsen/SpanMarkerNER/blob/main/notebooks/getting_started.ipynb) |
[📄 Documentation](https://tomaarsen.github.io/SpanMarkerNER)
</div>

SpanMarker is a framework for training powerful Named Entity Recognition models using familiar encoders such as BERT, RoBERTa and DeBERTa.
Tightly implemented on top of the [🤗 Transformers](https://github.com/huggingface/transformers/) library, SpanMarker can take advantage of its valuable functionality.
<!-- like performance dashboard integration, automatic mixed precision, 8-bit inference-->

Based on the [PL-Marker](https://arxiv.org/pdf/2109.06067.pdf) paper, SpanMarker breaks the mold through its accessibility and ease of use. Crucially, SpanMarker works out of the box with many common encoders such as `bert-base-cased` and `roberta-large`, and automatically works with datasets using the `IOB`, `IOB2`, `BIOES`, `BILOU` or no label annotation scheme.

## Documentation
Feel free to have a look at the [documentation](https://tomaarsen.github.io/SpanMarkerNER).

## Installation
You may install the [`span_marker`](https://pypi.org/project/span-marker) Python module via `pip` like so:
```
pip install span_marker
```

## Quick Start
Please have a look at our [Getting Started](notebooks/getting_started.ipynb) notebook for details on how SpanMarker is commonly used. It explains the following snippet in more detail.

| Colab                                                                                                                                                                                                         | Kaggle                                                                                                                                                                                                             | Gradient                                                                                                                                                                                         | Studio Lab                                                                                                                                                                                                             |
|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tomaarsen/SpanMarkerNER/blob/main/notebooks/getting_started.ipynb)                       | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/tomaarsen/SpanMarkerNER/blob/main/notebooks/getting_started.ipynb)                       | [![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/tomaarsen/SpanMarkerNER/blob/main/notebooks/getting_started.ipynb)                       | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/tomaarsen/SpanMarkerNER/blob/main/notebooks/getting_started.ipynb)                       |

```python
from datasets import load_dataset
from span_marker import SpanMarkerModel, Trainer
from transformers import TrainingArguments

def main():
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
        dataloader_num_workers=2,
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

if __name__ == "__main__":
    main()
```

<!-- Because this work is based on [PL-Marker](https://arxiv.org/pdf/2109.06067v5.pdf), you may expect similar results to its [Papers with Code Leaderboard](https://paperswithcode.com/paper/pack-together-entity-and-relation-extraction) results. -->

## Pretrained Models

* [`tomaarsen/span-marker-bert-base-fewnerd-fine-super`](https://huggingface.co/tomaarsen/span-marker-bert-base-fewnerd-fine-super) is a model that I have trained in 2 hours on the finegrained, supervised [Few-NERD dataset](https://huggingface.co/datasets/DFKI-SLT/few-nerd). It reached a 0.7053 Test F1, competitive in the all-time [Few-NERD leaderboard](https://paperswithcode.com/sota/named-entity-recognition-on-few-nerd-sup) using `bert-base`. My training script resembles the one that you can see above.
  * Try the model out online using this [🤗 Space](https://tomaarsen-span-marker-bert-base-fewnerd-fine-super.hf.space/).

* [`tomaarsen/span-marker-roberta-large-fewnerd-fine-super`](https://huggingface.co/tomaarsen/span-marker-roberta-large-fewnerd-fine-super) was trained in 6 hours on the finegrained, supervised [Few-NERD dataset](https://huggingface.co/datasets/DFKI-SLT/few-nerd) using `roberta-large`. It reached a 0.7103 Test F1, very competitive in the all-time [Few-NERD leaderboard](https://paperswithcode.com/sota/named-entity-recognition-on-few-nerd-sup).

## Context
<h1 align="center">
    <a href="https://github.com/argilla-io/argilla">
    <img src="https://github.com/dvsrepo/imgs/raw/main/rg.svg" alt="Argilla" width="150">
    </a>
</h1>

I have developed this library as a part of my thesis work at [Argilla](https://github.com/argilla-io/argilla).
Feel free to ⭐ star or watch the SpanMarker repository to get notified when my thesis is published.

## Changelog
See [CHANGELOG.md](CHANGELOG.md) for news on all SpanMarker versions.

## License
See [LICENSE](LICENSE.md) for the current license.
