.. SpanMarker documentation master file, created by
   sphinx-quickstart on Thu Apr  6 09:03:27 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

#######################################
SpanMarker for Named Entity Recognition
#######################################

SpanMarker is a framework for training powerful Named Entity Recognition models using familiar encoders such as BERT,
RoBERTa and ELECTRA. Tightly implemented on top of the `🤗 Transformers <https://github.com/huggingface/transformers>`_
library, SpanMarker can take advantage of its valuable functionality.

Based on the `PL-Marker <https://arxiv.org/pdf/2109.06067.pdf>`_ paper, SpanMarker breaks the mold through its
accessibility and ease of use. Crucially, SpanMarker works out of the box with many common encoders such as
`bert-base-cased` and `roberta-large`, and automatically works with datasets using the `IOB`, `IOB2`, `BIOES`, `BILOU`
or no label annotation scheme.

Check out all publicly available SpanMarker models on the Hugging Face Hub `here <https://huggingface.co/models?library=span-marker>`_.
Alternatively, check out any model from this list of particularly useful models:

+-----------------------------------------------------------------------------------------------------------------------------------------------------+------------+-------------+--------------+
| Model ID                                                                                                                                            | Domain     | Label Count | Language     |
+=====================================================================================================================================================+============+=============+==============+
| `tomaarsen/span-marker-mbert-base-multinerd <https://huggingface.co/tomaarsen/span-marker-mbert-base-multinerd>`_                                   | General    | 15          | Multilingual |
|                                                                                                                                                     |            |             |              |
| `lxyuan/span-marker-bert-base-multilingual-uncased-multinerd <https://huggingface.co/lxyuan/span-marker-bert-base-multilingual-uncased-multinerd>`_ |            |             |              |
|                                                                                                                                                     |            |             |              |
| `lxyuan/span-marker-bert-base-multilingual-cased-multinerd <https://huggingface.co/lxyuan/span-marker-bert-base-multilingual-cased-multinerd>`_     |            |             |              |
+-----------------------------------------------------------------------------------------------------------------------------------------------------+------------+-------------+--------------+
| `tomaarsen/span-marker-bert-base-fewnerd-fine-super <https://huggingface.co/tomaarsen/span-marker-bert-base-fewnerd-fine-super>`_                   | General    | 66          | English      |
|                                                                                                                                                     |            |             |              |
| `tomaarsen/span-marker-xlm-roberta-base-fewnerd-fine-super <https://huggingface.co/tomaarsen/span-marker-xlm-roberta-base-fewnerd-fine-super>`_     |            |             | Multilingual |
+-----------------------------------------------------------------------------------------------------------------------------------------------------+------------+-------------+--------------+
| `tomaarsen/span-marker-bert-base-cross-ner <https://huggingface.co/tomaarsen/span-marker-bert-base-cross-ner>`_                                     | General    | 39          | English      |
|                                                                                                                                                     |            |             |              |
| `tomaarsen/span-marker-bert-base-uncased-cross-ner <https://huggingface.co/tomaarsen/span-marker-bert-base-uncased-cross-ner>`_                     |            |             |              |
+-----------------------------------------------------------------------------------------------------------------------------------------------------+------------+-------------+--------------+
| `tomaarsen/span-marker-roberta-large-ontonotes5 <https://huggingface.co/tomaarsen/span-marker-roberta-large-ontonotes5>`_                           | General    | 18          | English      |
+-----------------------------------------------------------------------------------------------------------------------------------------------------+------------+-------------+--------------+
| `tomaarsen/span-marker-bert-base-uncased-keyphrase-inspec <https://huggingface.co/tomaarsen/span-marker-bert-base-uncased-keyphrase-inspec>`_       | Keyphrases | 1           | English      |
+-----------------------------------------------------------------------------------------------------------------------------------------------------+------------+-------------+--------------+
| `tomaarsen/span-marker-bert-base-acronyms <https://huggingface.co/tomaarsen/span-marker-bert-base-acronyms>`_                                       | Acronyms   | 2           | English      |
|                                                                                                                                                     |            |             |              |
| `tomaarsen/span-marker-bert-base-uncased-acronyms <https://huggingface.co/tomaarsen/span-marker-bert-base-uncased-acronyms>`_                       |            |             |              |
+-----------------------------------------------------------------------------------------------------------------------------------------------------+------------+-------------+--------------+
| `tomaarsen/span-marker-bert-base-ncbi-disease <https://huggingface.co/tomaarsen/span-marker-bert-base-ncbi-disease>`_                               | Biomedical | 1           | English      |
|                                                                                                                                                     |            |             |              |
| `tomaarsen/span-marker-bert-base-uncased-bionlp <https://huggingface.co/tomaarsen/span-marker-bert-base-uncased-bionlp>`_                           |            | 5           |              |
+-----------------------------------------------------------------------------------------------------------------------------------------------------+------------+-------------+--------------+
| `stefan-it/span-marker-gelectra-large-germeval14 <https://huggingface.co/stefan-it/span-marker-gelectra-large-germeval14>`_                         | General    | 12          | German       |
|                                                                                                                                                     |            |             |              |
| `gwlms/span-marker-teams-germeval14 <https://huggingface.co/gwlms/span-marker-teams-germeval14>`_                                                   |            |             |              |
|                                                                                                                                                     |            |             |              |
| `gwlms/span-marker-token-dropping-bert-germeval14 <https://huggingface.co/gwlms/span-marker-token-dropping-bert-germeval14>`_                       |            |             |              |
|                                                                                                                                                     |            |             |              |
| `gwlms/span-marker-bert-germeval14 <https://huggingface.co/gwlms/span-marker-bert-germeval14>`_                                                     |            |             |              |
+-----------------------------------------------------------------------------------------------------------------------------------------------------+------------+-------------+--------------+


Context
=======

.. raw:: html

   <h1 align="center">
      <a href="https://github.com/argilla-io/argilla">
      <img src="https://github.com/dvsrepo/imgs/raw/main/rg.svg" alt="Argilla" width="150">
      </a>
   </h1>

I have developed this library as a part of my thesis work at `Argilla <https://github.com/argilla-io/argilla>`_.
Feel free to ⭐ star or watch the `SpanMarker repository <https://github.com/tomaarsen/SpanMarkerNER>`_ to get notified when my thesis is published.

***************
Quick Reference
***************

How to Train
============

::

   from datasets import load_dataset
   from span_marker import SpanMarkerModel, Trainer
   from transformers import TrainingArguments

   # The dataset labels can have a tagging schemed (IOB, IOB2, BIOES),
   # but that is not necessary. This dataset has no tagging scheme:
   dataset = load_dataset("DFKI-SLT/few-nerd", "supervised")
   labels = ["O", "art", "building", "event", "location", "organization", "other", "person", "product"]

   # Initialize a SpanMarkerModel using an encoder, e.g. BERT:
   model_name = "bert-base-cased"
   model = SpanMarkerModel.from_pretrained(model_name, labels=labels)

   # See the 🤗 TrainingArguments documentation for details here
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
       warmup_ratio=0.1,
   )

   # Our Trainer subclasses the 🤗 Trainer, and the usage is very similar
   trainer = Trainer(
       model=model,
       args=args,
       train_dataset=dataset["train"].select(range(8000)),
       eval_dataset=dataset["validation"].select(range(2000)),
   )

   # Training is really simple using our Trainer!
   trainer.train()
   trainer.save_model("my_span_marker_model/checkpoint-final")

   # ... and so is evaluating!
   metrics = trainer.evaluate()
   print(metrics)

How to predict
==============

::

   from span_marker import SpanMarkerModel

   # Load a finetuned SpanMarkerModel from the 🤗 Hub
   model = SpanMarkerModel.from_pretrained("tomaarsen/span-marker-bert-base-fewnerd-fine-super")

   # It is recommended to explicitly move the model to CUDA for faster inference
   model.cuda()

   model.predict("A prototype was fitted in the mid-'60s in a one-off DB5 extended 4'' after the doors and driven by Marek personally, and a normally 6-cylinder Aston Martin DB7 was equipped with a V8 unit in 1998.")
   [{'span': 'DB5', 'label': 'product-car', 'score': 0.8675689101219177, 'char_start_index': 52, 'char_end_index': 55},
    {'span': 'Marek', 'label': 'person-other', 'score': 0.9100819230079651, 'char_start_index': 99, 'char_end_index': 104},
    {'span': 'Aston Martin DB7', 'label': 'product-car', 'score': 0.9931442737579346, 'char_start_index': 143, 'char_end_index': 159}]

.. note::
   You can also load a locally saved model through ``SpanMarkerModel.from_pretrained("path/to/model")``,
   much like in 🤗 Transformers.


How to save a model
===================

Locally
-------

::

   model.save_pretrained("my_model_dir")


To the 🤗 Hub
-------------

::

   model_name = "span-marker-bert-base-fewnerd-fine-super"
   model.push_to_hub(model_name)


.. toctree::
   :maxdepth: 2
   :caption: Documentation
   :titlesonly:

   Notebooks <notebooks/index>
   API Reference <api/span_marker>

.. toctree::
   :maxdepth: 1
   :hidden:
   :titlesonly:
   :caption: Installation

   install

.. toctree::
   :maxdepth: 1
   :hidden:
   :titlesonly:
   :caption: More

   news
   Open Issues <https://github.com/tomaarsen/SpanMarkerNER/issues>
   SpanMarker on GitHub <https://github.com/tomaarsen/SpanMarkerNER>
   SpanMarker on the 🤗 Hub <https://huggingface.co/models?library=span-marker>
   SpanMarker on spaCy <https://spacy.io/universe/project/span_marker>
