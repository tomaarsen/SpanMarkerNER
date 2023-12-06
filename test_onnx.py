from span_marker.onnx.spanmarker_onnx import export_spanmarker_to_onnx, SpanMarkerOnnx
from span_marker.tokenizer import SpanMarkerTokenizer, SpanMarkerConfig

repo_id = "lxyuan/span-marker-bert-base-multilingual-uncased-multinerd"
export_spanmarker_to_onnx(repo_id)
config = SpanMarkerConfig.from_pretrained(repo_id)
tokenizer = SpanMarkerTokenizer.from_pretrained(repo_id, config=config)
spanmarker_tokenizer = SpanMarkerTokenizer.from_pretrained(repo_id, config=config)

model = SpanMarkerOnnx(
    onnx_encoder_path="spanmarker_encoder.onnx",
    onnx_classifier_path="spanmarker_classifier.onnx",
    tokenizer=spanmarker_tokenizer,
    config=config,
)
print(model.predict(["antonio polo de alvarado vive en España"] * 10))
