from datasets import load_dataset
from transformers import TrainingArguments

from span_marker import SpanMarkerModel, SpanMarkerModelCardData, Trainer


def main() -> None:
    # Load the dataset, ensure "tokens" and "ner_tags" columns, and get a list of labels
    dataset_id = "tner/ontonotes5"
    dataset_name = "OntoNotes v5"
    dataset = load_dataset(dataset_id)
    dataset = dataset.rename_column("tags", "ner_tags")
    labels = [
        "O",
        "B-CARDINAL",
        "B-DATE",
        "I-DATE",
        "B-PERSON",
        "I-PERSON",
        "B-NORP",
        "B-GPE",
        "I-GPE",
        "B-LAW",
        "I-LAW",
        "B-ORG",
        "I-ORG",
        "B-PERCENT",
        "I-PERCENT",
        "B-ORDINAL",
        "B-MONEY",
        "I-MONEY",
        "B-WORK_OF_ART",
        "I-WORK_OF_ART",
        "B-FAC",
        "B-TIME",
        "I-CARDINAL",
        "B-LOC",
        "B-QUANTITY",
        "I-QUANTITY",
        "I-NORP",
        "I-LOC",
        "B-PRODUCT",
        "I-TIME",
        "B-EVENT",
        "I-EVENT",
        "I-FAC",
        "B-LANGUAGE",
        "I-PRODUCT",
        "I-ORDINAL",
        "I-LANGUAGE",
    ]

    # Initialize a SpanMarker model using a pretrained BERT-style encoder
    encoder_id = "roberta-large"
    model = SpanMarkerModel.from_pretrained(
        encoder_id,
        labels=labels,
        # SpanMarker hyperparameters:
        model_max_length=256,
        marker_max_length=128,
        entity_max_length=10,
        # Model card arguments
        model_card_data=SpanMarkerModelCardData(
            model_id=f"tomaarsen/span-marker-{encoder_id}-ontonotes5",
            encoder_id=encoder_id,
            dataset_name=dataset_name,
            dataset_id=dataset_id,
            license="other",
            language="en",
        ),
    )

    # Prepare the 🤗 transformers training arguments
    args = TrainingArguments(
        output_dir="models/span_marker_roberta_large_ontonotes5",
        # Training Hyperparameters:
        learning_rate=1e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=4,
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
    trainer.save_model("models/span_marker_roberta_large_ontonotes5/checkpoint-final")

    # Compute & save the metrics on the test set
    metrics = trainer.evaluate(dataset["test"], metric_key_prefix="test")
    trainer.save_metrics("test", metrics)


if __name__ == "__main__":
    main()
