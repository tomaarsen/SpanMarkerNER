from typing import Any, Dict, List, Tuple, Union

from transformers import Pipeline

INPUT_TYPES = Union[str, List[str], List[List[str]]]
OUTPUT_TYPES = Union[List[Dict[str, Union[str, int, float]]], List[List[Dict[str, Union[str, int, float]]]]]


class SpanMarkerPipeline(Pipeline):
    """A Pipeline component for SpanMarker.

    The `pipeline` function is :func:`~transformers.pipeline`, which you can also import with
    ``from transformers import pipeline``, but you must also import ``span_marker`` to register the
    ``"span-marker"`` pipeline task.

    Example::

        >>> from span_marker import pipeline
        >>> pipe = pipeline(task="span-marker", model="tomaarsen/span-marker-mbert-base-multinerd", device_map="auto")
        >>> pipe("Amelia Earhart flew her single engine Lockheed Vega 5B across the Atlantic to Paris.")
        [{'span': 'Amelia Earhart', 'label': 'PER', 'score': 0.9999709129333496, 'char_start_index': 0, 'char_end_index': 14},
         {'span': 'Lockheed Vega 5B', 'label': 'VEHI', 'score': 0.9050095677375793, 'char_start_index': 38, 'char_end_index': 54},
         {'span': 'Atlantic', 'label': 'LOC', 'score': 0.9991973042488098, 'char_start_index': 66, 'char_end_index': 74},
         {'span': 'Paris', 'label': 'LOC', 'score': 0.9999232292175293, 'char_start_index': 78, 'char_end_index': 83}]

    """

    def _sanitize_parameters(self, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        return {}, {}, {}

    def preprocess(self, inputs: INPUT_TYPES) -> INPUT_TYPES:
        return inputs

    def _forward(self, inputs: INPUT_TYPES) -> OUTPUT_TYPES:
        return self.model.predict(inputs)

    def postprocess(self, outputs: OUTPUT_TYPES) -> OUTPUT_TYPES:
        return outputs
