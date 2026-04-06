from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from decision_tree_part import run_decision_tree_experiments
from heuristic_search import run_heuristic_search_experiments


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    outputs_root = root / "outputs"

    heuristic_dir = outputs_root / "heuristic_search"
    decision_tree_dir = outputs_root / "decision_tree"

    heuristic_summary = run_heuristic_search_experiments(heuristic_dir)
    tree_summary = run_decision_tree_experiments(decision_tree_dir)

    final_summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "parts": {
            "heuristic_search": heuristic_summary,
            "decision_tree": tree_summary,
        },
    }

    with (outputs_root / "execution_summary.json").open("w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2)

    print("Execution complete. Artifacts saved in:")
    print(f"- {heuristic_dir}")
    print(f"- {decision_tree_dir}")
    print(f"- {outputs_root / 'execution_summary.json'}")


if __name__ == "__main__":
    main()
