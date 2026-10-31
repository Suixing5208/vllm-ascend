# Efficient Video Sampling (EVS)

Ascend Qwen3.5 dense and MoE models support the upstream EVS video-token
pruning path. Enable it with a positive `--video-pruning-rate`:

```bash
vllm serve /path/to/Qwen3.5-2B \
  --limit-mm-per-prompt.video 1 \
  --mm-processor-kwargs '{"fps": 1}' \
  --video-pruning-rate 0.75
```

The rate is the fraction of video tokens targeted for pruning. Text and video
separator tokens remain in the prompt, so the total prompt reduction can be
smaller than the configured rate. Without this option, the normal multimodal
path is used.

EVS is available only when the selected model and processor expose the
upstream multimodal-pruning interface. Validate output quality on the target
video workload before changing the pruning rate.
