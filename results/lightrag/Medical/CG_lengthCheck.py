import json
import numpy as np

def get_missing_faith(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    scores = data["Creative Generation"]["per_sample_scores"]
    acc_ids   = set(scores.get("answer_correctness", {}).keys())
    faith_ids = set(scores.get("faithfulness", {}).keys())
    return acc_ids - faith_ids

always_missing = (get_missing_faith("evaluation_qwen3_perSample.json")
                & get_missing_faith("evaluation_reversed_perSample.json")
                & get_missing_faith("evaluation_reversed_sameDes_perSample.json"))

# Load all three predictions files
pred_files = {
    "baseline": "predictions_Medical_qwen3.json",
    #"revert":   "predictions_Medical_qwen3_revert.json",
    "ablation": "test_predictions_Medical_reversed_sameDes.json",
}

preds = {}
for condition, path in pred_files.items():
    with open(path, encoding="utf-8") as f:
        preds[condition] = {d["id"]: d for d in json.load(f)}

# Report answer and context lengths for always-missing samples across all conditions
print(f"{'ID':<25} {'Condition':<12} {'answer_len':>12} {'context_len':>13}")
print("-" * 65)
for sid in sorted(always_missing):
    for condition, pred_dict in preds.items():
        item = pred_dict.get(sid, {})
        ans = item.get("generated_answer", "")
        ctx = item.get("context", "")
        print(f"{sid:<25} {condition:<12} {len(ans):>12,} {len(ctx):>13,}")
    print()

# Compare always-missing vs rest, across all conditions
print("\nSummary: mean answer_len across all conditions")
print(f"{'Condition':<12} {'always_missing mean':>20} {'rest mean':>12}")
print("-" * 46)
for condition, pred_dict in preds.items():
    cg_items = [d for d in pred_dict.values()]
    missing_lens = [len(d.get("generated_answer",""))
                    for d in cg_items if d["id"] in always_missing]
    rest_lens    = [len(d.get("generated_answer",""))
                    for d in cg_items if d["id"] not in always_missing]
    print(f"{condition:<12} {np.mean(missing_lens):>20.0f} {np.mean(rest_lens):>12.0f}")