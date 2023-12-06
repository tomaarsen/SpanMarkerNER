from span_marker.onnx.spanmarker_onnx import export_spanmarker_to_onnx, SpanMarkerOnnx
from span_marker.tokenizer import SpanMarkerTokenizer, SpanMarkerConfig
import time
from span_marker import SpanMarkerModel

def run_test(base_model:SpanMarkerModel,onnx_model:SpanMarkerOnnx,sample:str="Huggingface is the best AI company",batch_size=1,reps=1):
    """
    Runs a performance test comparing a SpanMarkerModel and its ONNX equivalent, SpanMarkerOnnx.

    This function is used to evaluate and compare the inference time and results consistency between a PyTorch implementation
    (SpanMarkerModel) and an ONNX implementation (SpanMarkerOnnx) of the SpanMarker model. It processes a given sample multiple times
    (defined by 'reps') and calculates the average inference time for each model.

    Args:
        base_model (SpanMarkerModel): 
            The SpanMarkerModel instance to be tested.
        onnx_model (SpanMarkerOnnx): 
            The SpanMarkerOnnx instance to be tested.
        sample (str, optional): 
            The sample text to be processed by the models. Defaults to "Huggingface is the best AI company".
        batch_size (int, optional): 
            The size of the batch to be processed by the models. Defaults to 1.
        reps (int, optional): 
            The number of repetitions for running the test to average out the inference time. Defaults to 1.

    The function measures the time taken for each repetition for both models, prints the average times, and checks if the results
    from both models are consistent (excluding the score values).
    """
    
    batch = [
        sample
    ] * batch_size

    base_model = SpanMarkerModel.from_pretrained(repo_id)
    torch_times = []
    onnx_times = []
    for _ in range(reps):
        start_time = time.time()
        torch_result = base_model.predict(batch, batch_size)
        end_time = time.time()
        torch_time = end_time - start_time
        torch_times.append(torch_time)

        start_time = time.time()
        onnx_result = onnx_model.predict(batch, batch_size=batch_size)
        end_time = time.time()
        onnx_time = end_time - start_time
        onnx_times.append(onnx_time)

    print(f"Time results:")
    print(f"Batch size: {len(batch)}")
    print(f"Torch times: {torch_times}")
    print(f"ONNX CPU times: {onnx_times}")
    print(f"Avg Torch time: {sum(torch_times)/len(torch_times)}")
    print(f"Avg ONNX CPU time: {sum(onnx_times)/len(onnx_times)}")

    def strip_score_from_results(results):
        return [[{key: value for key, value in ent.items() if key != "score"} for ent in ents] for ents in results]

    print(f"Results are the same: {strip_score_from_results(torch_result)==strip_score_from_results(onnx_result)}")


if __name__ == "__main__":

    # Introduce your repo_id
    repo_id = "guishe/span-marker-generic-ner-v1-fewnerd-fine-super"

    # Export encoder and classifier to ONNX
    export_spanmarker_to_onnx(repo_id)

    # Get you SpanMarkerOnnx model
    config = SpanMarkerConfig.from_pretrained(repo_id)
    tokenizer = SpanMarkerTokenizer.from_pretrained(repo_id, config=config)
    spanmarker_tokenizer = SpanMarkerTokenizer.from_pretrained(repo_id, config=config)
    onnx_cpu = SpanMarkerOnnx(
        onnx_encoder_path="spanmarker_encoder.onnx",
        onnx_classifier_path="spanmarker_classifier.onnx",
        tokenizer=spanmarker_tokenizer,
        config=config,
    )

    # Base Model VS Onnx Model
    base_model = SpanMarkerModel.from_pretrained("lxyuan/span-marker-bert-base-multilingual-uncased-multinerd")
    sample = "Huggingface is the best AI company in the world"
    batch_size = 5
    reps = 1
    run_test(base_model=base_model,onnx_model=onnx_cpu,sample=sample,batch_size=batch_size,reps=reps)
