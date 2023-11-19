import sys
from pathlib import Path
import time
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))
from span_marker import SpanMarkerModel

repo_id = "lxyuan/span-marker-bert-base-multilingual-uncased-multinerd"
model = SpanMarkerModel.from_pretrained(repo_id)

batch_size = 10
batch = ["Antonio Polo is living in Cáceres","Antonio Polo is living in Cáceres","Antonio Polo is living in Cáceres","Antonio Polo is living in Cáceres"]* batch_size

number_of_tests = 10
time_results = []
results = []

for _ in range(number_of_tests):
    start_time = time.time()
    torch_result = model.predict(batch, batch_size)
    end_time = time.time()
    torch_time = end_time - start_time
    time_results.append(torch_time)
    results.append(torch_result)
print(f"Mean time: {sum(time_results)/len(time_results)}")

