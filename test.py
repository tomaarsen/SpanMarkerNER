import sys
from pathlib import Path
import time
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))
from span_marker import SpanMarkerModel

repo_id = "lxyuan/span-marker-bert-base-multilingual-uncased-multinerd"
model = SpanMarkerModel.from_pretrained(repo_id)
model = torch.compile(model)

batch_size = 100
batch = [
    "Pedro is working in Alicante. Pedro is working in Alicante. Pedro is working in Alicante.Pedro is working in Alicante. Pedro is working in Alicante. Pedro is working in Alicante.Pedro is working in Alicante. Pedro is working in Alicante. Pedro is working in Alicante",
] * batch_size


print(f"-------- Start Torch--------")
start_time = time.time()
torch_result = model.predict(batch, batch_size)
end_time = time.time()
torch_time = end_time - start_time
print(f"-------- End Torch --------")
print(f"Time results:")
print(f"Batch size: {len(batch)}")
print(f"Torch time: {torch_time}")
