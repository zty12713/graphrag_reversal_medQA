import json

def get_faith_ids(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    scores = data["Creative Generation"]["per_sample_scores"]
    acc_ids   = set(scores.get("answer_correctness", {}).keys())
    faith_ids = set(scores.get("faithfulness", {}).keys())
    return acc_ids - faith_ids   # samples missing faithfulness

missing_baseline  = get_faith_ids("evaluation_qwen3_perSample.json")
missing_revert    = get_faith_ids("evaluation_reversed_perSample.json")
missing_ablation  = get_faith_ids("evaluation_reversed_sameDes_perSample.json")

print(f"Missing in baseline:  {len(missing_baseline)}")
print(f"Missing in revert:    {len(missing_revert)}")
print(f"Missing in ablation:  {len(missing_ablation)}")

always_missing = missing_baseline & missing_revert & missing_ablation
never_missing  = len(missing_baseline | missing_revert | missing_ablation)

print(f"\nMissing in ALL three conditions: {len(always_missing)}")
print(f"Missing in at least one condition: {never_missing}")