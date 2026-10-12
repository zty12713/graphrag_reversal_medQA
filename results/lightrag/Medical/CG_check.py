import json

file = "evaluation_reversed_sameDes_perSample.json"
with open(file, encoding="utf-8") as f:
    data = json.load(f)

cg_scores = data["Creative Generation"]["per_sample_scores"]

# Which metrics have scores for which samples
faith_ids   = set(cg_scores.get("faithfulness", {}).keys())
acc_ids     = set(cg_scores.get("answer_correctness", {}).keys())

missing_faith = acc_ids - faith_ids   # samples with accuracy but no faithfulness
print(f"Samples with answer_correctness but no faithfulness: {len(missing_faith)}")

# Check context lengths for missing vs present samples
with open("predictions_Medical_qwen3.json", encoding="utf-8") as f:
    predictions = json.load(f)

cg_preds = {d["id"]: d for d in predictions
            if d.get("question_type") == "Creative Generation"}

lengths_missing = [len(str(cg_preds[sid].get("context", "")))
                   for sid in missing_faith if sid in cg_preds]
lengths_present = [len(str(cg_preds[sid].get("context", "")))
                   for sid in faith_ids if sid in cg_preds]

import numpy as np
print(f"\nContext length — missing faithfulness: mean={np.mean(lengths_missing):.0f}, max={np.max(lengths_missing):.0f}")
print(f"Context length — has faithfulness:     mean={np.mean(lengths_present):.0f}, max={np.max(lengths_present):.0f}")