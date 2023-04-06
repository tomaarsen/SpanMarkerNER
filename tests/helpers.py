from typing import Dict, List, Union


def compare_entities(
    pred_entities: List[Dict[str, Union[str, float, int]]],
    gold_entities: List[Dict[str, Union[str, int]]],
) -> None:
    """Compare a list of prediction entities to a list of gold entities, with exception of the 'score' values.

    Args:
        pred_entities (List[Dict[str, Union[str, float, int]]]): List of prediction entities from `model.predict(...)`.
        gold_entities (List[Dict[str, Union[str, int]]]): List of gold entities.
    """
    # Ensure the same length
    assert len(gold_entities) == len(pred_entities)
    for pred, gold in zip(pred_entities, gold_entities):
        # ... and keys
        assert set(pred.keys()) == set(gold.keys()) | {"score"}
        for key, value in gold.items():
            # ... and values
            assert pred[key] == value
