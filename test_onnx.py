from span_marker.onnx.spanmarker_onnx import export_spanmarker_to_onnx, SpanMarkerOnnx
from span_marker.tokenizer import SpanMarkerTokenizer, SpanMarkerConfig

repo_id = "guishe/span-marker-generic-ner-v1-fewnerd-fine-super"
export_spanmarker_to_onnx(repo_id)
config = SpanMarkerConfig.from_pretrained(repo_id)
tokenizer = SpanMarkerTokenizer.from_pretrained(repo_id, config=config)
spanmarker_tokenizer = SpanMarkerTokenizer.from_pretrained(repo_id, config=config)

onnx_cpu = SpanMarkerOnnx(
    onnx_encoder_path="spanmarker_encoder.onnx",
    onnx_classifier_path="spanmarker_classifier.onnx",
    tokenizer=spanmarker_tokenizer,
    config=config,
)


print(onnx_cpu.predict(["Antonio Polo de Alvarado lives in Spain"] * 10))
