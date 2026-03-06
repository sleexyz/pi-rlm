"""Custom reward function for ARC-AGI-2 SDPO training.

For multi-turn ARC, the reward is already computed by ArcInteraction
(1.0 for correct submission, 0.0 otherwise). This function extracts
the turn-level reward from the interaction's accumulated scores.

verl calls this as a fallback when rm_scores isn't in the batch.
"""

import json


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
) -> dict:
    """Compute reward for a completed rollout.

    For arc-agi-2: the reward is based on whether the model's submission
    matches the expected test output. Since multi-turn interactions accumulate
    turn_scores, this serves as a final scoring pass.
    """
    if data_source == "arc-agi-2":
        # Parse ground truth (expected output grid)
        try:
            expected = json.loads(ground_truth)
        except (json.JSONDecodeError, TypeError):
            return {"score": 0.0, "is_correct": False}

        # Try to extract submitted grid from the solution
        # In multi-turn, solution_str is the full conversation text
        # The actual reward is computed by the interaction's turn_scores
        # Return 0.0 as base — SDPO uses turn_scores for the advantage
        return {"score": 0.0, "is_correct": False}

    raise ValueError(f"Unknown data source: {data_source}")
