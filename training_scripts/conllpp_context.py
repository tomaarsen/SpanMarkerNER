from datasets import load_dataset
from transformers import TrainingArguments

from span_marker import SpanMarkerModel, Trainer


def main() -> None:
    # Load the dataset, ensure "tokens", "ner_tags", "document_id" and "sentence_id" columns,
    # and get a list of labels
    dataset = load_dataset("tomaarsen/conllpp")
    labels = dataset["train"].features["ner_tags"].feature.names

    # Initialize a SpanMarker model using a pretrained BERT-style encoder
    model_name = "xlm-roberta-large"
    model = SpanMarkerModel.from_pretrained(
        model_name,
        labels=labels,
        # SpanMarker hyperparameters:
        model_max_length=512,
        marker_max_length=128,
        entity_max_length=8,
    )

    # Prepare the 🤗 transformers training arguments
    args = TrainingArguments(
        output_dir="models/span_marker_xlm_roberta_large_conllpp_doc_context",
        # Training Hyperparameters:
        learning_rate=1e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        weight_decay=0.01,
        warmup_ratio=0.1,
        bf16=True,  # Replace `bf16` with `fp16` if your hardware can't use bf16.
        # Other Training parameters
        logging_first_step=True,
        logging_steps=50,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=1000,
        save_total_limit=2,
        dataloader_num_workers=2,
    )

    # Initialize the trainer using our model, training args & dataset, and train
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )
    trainer.train()
    trainer.save_model("models/span_marker_xlm_roberta_large_conllpp_doc_context/checkpoint-final")

    # Compute & save the metrics on the test set
    metrics = trainer.evaluate(dataset["test"], metric_key_prefix="test")
    trainer.save_metrics("test", metrics)


if __name__ == "__main__":
    main()
