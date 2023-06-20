from setuptools import setup

setup(entry_points={"spacy_factories": ["span_marker = span_marker.__init__:_spacy_span_marker_factory"]})
