from datasets import load_dataset
from transformers import TrainingArguments

from span_marker import SpanMarkerModel, SpanMarkerModelCardData, Trainer


def main() -> None:
    # Load the dataset, ensure "tokens" and "ner_tags" columns, and get a list of labels
    dataset_id = "DFKI-SLT/few-nerd"
    dataset_name = "FewNERD"
    dataset = load_dataset(dataset_id, "supervised")
    dataset = dataset.remove_columns("ner_tags")
    dataset = dataset.rename_column("fine_ner_tags", "ner_tags")
    labels = dataset["train"].features["ner_tags"].feature.names

    # Initialize a SpanMarker model using a pretrained BERT-style encoder
    encoder_id = "bert-base-cased"
    model = SpanMarkerModel.from_pretrained(
        encoder_id,
        labels=labels,
        # SpanMarker hyperparameters:
        model_max_length=256,
        marker_max_length=128,
        entity_max_length=8,
        # Model card arguments
        model_card_data=SpanMarkerModelCardData(
            model_id="tomaarsen/span-marker-bert-base-fewnerd-fine-super",
            encoder_id=encoder_id,
            dataset_name=dataset_name,
            dataset_id=dataset_id,
            license="cc-by-nc-sa-4.0",
            language="en",
        ),
    )

    # Prepare the 🤗 transformers training arguments
    args = TrainingArguments(
        output_dir="models/span_marker_bert_base_cased_fewnerd_fine_super",
        # Training Hyperparameters:
        learning_rate=5e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        warmup_ratio=0.1,
        bf16=True,  # Replace `bf16` with `fp16` if your hardware can't use bf16.
        # Other Training parameters
        logging_first_step=True,
        logging_steps=50,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=3000,
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
    trainer.save_model("models/span_marker_bert_base_cased_fewnerd_fine_super/checkpoint-final")

    # Compute & save the metrics on the test set
    metrics = trainer.evaluate(dataset["test"], metric_key_prefix="test")
    trainer.save_metrics("test", metrics)


if __name__ == "__main__":
    main()
