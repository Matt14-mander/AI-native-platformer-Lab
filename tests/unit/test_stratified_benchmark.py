from ai_platformer.benchmark.stratified import summarize


def test_summarize_distinguishes_success_from_collection_target():
    rows = [
        {"outcome": "success", "collection_target_met": False, "progress": 1.0,
         "coin_ratio": 0.875, "steps": 200, "return": 12.0},
        {"outcome": "success", "collection_target_met": True, "progress": 1.0,
         "coin_ratio": 1.0, "steps": 210, "return": 13.0},
    ]
    result = summarize(rows)
    assert result["successes"] == 2
    assert result["collection_targets_met"] == 1
    assert result["mean_coin_ratio"] == 0.9375
