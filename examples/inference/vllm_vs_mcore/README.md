# PERFORMANCE SCRIPT
This will help compare mcore and vllm inference in a very simple setup. 

To run this on slurm 


## NEW: SIMPLE MCORE API (vLLM-like)

MCore now provides a vLLM-like API for easy migration. Here's a comparison:

### vLLM API:
```python
from vllm import LLM, SamplingParams

llm = LLM(model='Qwen/Qwen2.5-1.5B', tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.8, max_tokens=256)
outputs = llm.generate(["Hello, world!"], sampling_params)
```

### MCore API (NEW):
```python
from megatron.core.inference import LLM, SamplingParams

llm = LLM(model='/path/to/checkpoint', tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.8, max_tokens=256)
outputs = llm.generate(["Hello, world!"], sampling_params)
```

See `mcore_simple_inference.py` for a complete example using the new API.

##  RUNNING MCORE ON SLURM (Traditional Method)
```
CONTAINER_IMAGE=<container_image>

srun --gpus-per-node 8 --time 04:00:00  --account coreai_dlalgo_llm --job-name coreai_dlalgo_llm:inference --partition interactive --container-image $CONTAINER_IMAGE --container-mounts /lustre/:/lustre/ --no-container-mount-home --pty /bin/bash

cd /path/to/Megatron-LM/
export HF_TOKEN=<hf_token>

export NSIGHT_PREFIX=<path_to_nsight>

nsys profile -s none -t nvtx,cuda \
  --cudabacktrace=all \
  --cuda-graph-trace=node \
  --python-backtrace=cuda \
  --wait all \
  -o ${NSIGHT_PREFIX} \
  --force-overwrite true \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  torchrun --nproc-per-node 1 \
    -m examples.inference.performance_comparison.mcore_inference \
    --load /path/to/Qwen/Qwen2.5-1.5B \
    --bf16 \
    --tensor-model-parallel-size 1 \
    --micro-batch-size 64 \
    --enable-cuda-graph \
    --dist-ckpt-strictness log_unexpected \
    --decode-only-cuda-graphs \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model Qwen/Qwen2.5-1.5B \
    --no-use-tokenizer-model-from-checkpoint-args \
    --num-layers 28 \
    --hidden-size 1536 \
    --num-attention-heads 12 \
    --max-position-embeddings 32768 \
    --num-query-groups 2 \
    --group-query-attention \
    --swiglu \
    --normalization RMSNorm \
    --disable-bias-linear \
    --position-embedding-type rope \
    --rotary-percent 1.0 \
    --rotary-base 1000000 \
    --seq-length 32768 \
    --ffn-hidden-size 8960
```

##  RUNNING VLLM 
```
uv pip install vllm==0.11.0

python3 vllm_mcore_comparison/vllm_inference.py
```