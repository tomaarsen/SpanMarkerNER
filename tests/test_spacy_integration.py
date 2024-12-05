import spacy


def test_span_marker_as_spacy_pipeline_component():
    nlp = spacy.load("en_core_web_sm", exclude=["ner"])
    batch_size = 2
    wrapper = nlp.add_pipe(
        "span_marker", config={"model": "tomaarsen/span-marker-bert-tiny-conll03", "batch_size": batch_size}
    )
    assert wrapper.batch_size == batch_size

    doc = nlp("Amelia Earhart flew her single engine Lockheed Vega 5B across the Atlantic to Paris.")
    assert [(span.text, span.label_) for span in doc.ents] == [
        ("Amelia Earhart", "PER"),
        ("Lockheed Vega", "ORG"),
        ("Atlantic", "LOC"),
        ("Paris", "LOC"),
    ]

    # Override a setting that modifies how inference is performed,
    # should not have any impact with just one sentence, i.e. it should still work.
    wrapper.model.config.trained_with_document_context = True
    doc = nlp("Amelia Earhart flew her single engine Lockheed Vega 5B across the Atlantic to Paris.")
    assert [(span.text, span.label_) for span in doc.ents] == [
        ("Amelia Earhart", "PER"),
        ("Lockheed Vega", "ORG"),
        ("Atlantic", "LOC"),
        ("Paris", "LOC"),
    ]


def test_span_marker_as_spacy_pipeline_component_pipe():
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    batch_size = 2
    wrapper = nlp.add_pipe(
        "span_marker", config={"model": "tomaarsen/span-marker-bert-tiny-conll03", "batch_size": batch_size}
    )
    assert wrapper.batch_size == batch_size

    docs = nlp.pipe(["Amelia Earhart flew her single engine Lockheed Vega 5B across the Atlantic to Paris."])
    doc = list(docs)[0]
    assert [(span.text, span.label_) for span in doc.ents] == [
        ("Amelia Earhart", "PER"),
        ("Lockheed Vega", "ORG"),
        ("Atlantic", "LOC"),
        ("Paris", "LOC"),
    ]

    # Override a setting that modifies how inference is performed,
    # should not have any impact with just one sentence, i.e. it should still work.
    wrapper.model.config.trained_with_document_context = True
    docs = nlp.pipe(["Amelia Earhart flew her single engine Lockheed Vega 5B across the Atlantic to Paris."])
    doc = list(docs)[0]
    assert [(span.text, span.label_) for span in doc.ents] == [
        ("Amelia Earhart", "PER"),
        ("Lockheed Vega", "ORG"),
        ("Atlantic", "LOC"),
        ("Paris", "LOC"),
    ]
