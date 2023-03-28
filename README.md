# SpanMarkerNER

```python
from datasets import load_dataset
from span_marker import SpanMarkerModel, Trainer
from transformers import TrainingArguments

dataset = load_dataset("DFKI-SLT/few-nerd", "supervised")
labels = dataset["train"].features["ner_tags"].feature.names

model_name = "roberta-base"
model = SpanMarkerModel.from_pretrained(model_name, labels=labels, model_max_length=256)

args = TrainingArguments(
    output_dir="my_span_marker_model",
    learning_rate=1e-5,
    gradient_accumulation_steps=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    weight_decay=0.01,
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=200,
    push_to_hub=False,
    logging_first_step=True,
    logging_steps=50,
    bf16=True,
    dataloader_num_workers=0,
    warmup_ratio=0.1,
    seed=42,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"].select(range(9797)),
    eval_dataset=dataset["validation"].select(range(3107)),
)

trainer.train()
trainer.save_model(output_dir / "checkpoint-final")

metrics = trainer.evaluate()
print(metrics)
```