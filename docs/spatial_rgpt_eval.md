# Spatial-RGPT evaluation (Gemma baseline)

This document explains how to run the Gemma baseline on the Spatial-RGPT benchmark (no depth maps).

Dataset
-------
The Spatial-RGPT benchmark is available on HuggingFace at:

https://huggingface.co/datasets/a8cheng/SpatialRGPT-Bench

The eval script in this repo supports passing the HF dataset id directly or a local JSONL file.

Running the evaluation
----------------------

Example PowerShell command (runs on GPU if `--device cuda`):

```powershell
python .\scripts\eval_spatial_rgpt_gemma.py --data-path a8cheng/SpatialRGPT-Bench --model google/gemma-3-4b-it --device cuda --output .\outputs\spatial_rgpt_gemma_results.jsonl --max-samples 200
```

Notes
-----
- The script intentionally does NOT compute or use depth maps — it is a Gemma-only baseline for comparison with Spatial-RGPT (depth-based) results.
- If images are provided as URLs in the dataset, the loader will download them on-the-fly; network access is required.
- Set `--max-samples` to a small number to do a quick smoke test.

Next steps
----------
- After you generate results with this baseline, we can add a comparator that computes accuracy against Spatial-RGPT's outputs or create a PACE job script to run large-scale evaluation.
