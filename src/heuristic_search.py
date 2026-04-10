from __future__ import annotations

import csv
import heapq
import json
import math
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt

Point = Tuple[int, int]
Heuristic = Callable[[Point, Point], float]


@dataclass
class SearchResult:
    algorithm: str
    heuristic_name: str
    path: List[Point]
    path_cost: float
    expanded_nodes: int
    runtime_ms: float
    expansion_order: List[Point]
    tracked_scores: List[Dict[str, float]]
    peak_memory_kb: float = 0.0


class GridProblem:
    def __init__(
        self,
        width: int,
        height: int,
        start: Point,
        goal: Point,
        blocked: Iterable[Point],
        terrain_cost: Optional[Dict[Point, float]] = None,
    ) -> None:
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        self.blocked = set(blocked)
        self.terrain_cost = terrain_cost or {}

    def in_bounds(self, node: Point) -> bool:
        x, y = node
        return 0 <= x < self.width and 0 <= y < self.height

    def passable(self, node: Point) -> bool:
        return node not in self.blocked

    def neighbors(self, node: Point) -> List[Point]:
        x, y = node
        candidates = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        out = []
        for cand in candidates:
            if self.in_bounds(cand) and self.passable(cand):
                out.append(cand)
        return out

    def move_cost(self, node: Point) -> float:
        return self.terrain_cost.get(node, 1.0)


def manhattan(a: Point, b: Point) -> float:
    return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))


def euclidean(a: Point, b: Point) -> float:
    return math.dist(a, b)


def reconstruct_path(came_from: Dict[Point, Point], start: Point, goal: Point) -> List[Point]:
    if goal not in came_from and goal != start:
        return []
    node = goal
    path = [node]
    while node != start:
        node = came_from[node]
        path.append(node)
    path.reverse()
    return path


def greedy_best_first(problem: GridProblem, heuristic: Heuristic, heuristic_name: str) -> SearchResult:
    start_time = time.perf_counter()

    frontier: List[Tuple[float, Point]] = []
    heapq.heappush(frontier, (heuristic(problem.start, problem.goal), problem.start))

    came_from: Dict[Point, Point] = {}
    g_score: Dict[Point, float] = {problem.start: 0.0}
    visited = set()
    expansion_order: List[Point] = []
    tracked_scores: List[Dict[str, float]] = []

    while frontier:
        _, current = heapq.heappop(frontier)

        if current in visited:
            continue
        visited.add(current)

        expansion_order.append(current)
        tracked_scores.append(
            {
                "x": current[0],
                "y": current[1],
                "g": g_score[current],
                "h": heuristic(current, problem.goal),
                "f": heuristic(current, problem.goal),
            }
        )

        if current == problem.goal:
            break

        for nxt in problem.neighbors(current):
            if nxt in visited:
                continue
            step = problem.move_cost(nxt)
            new_cost = g_score[current] + step
            if nxt not in g_score or new_cost < g_score[nxt]:
                g_score[nxt] = new_cost
                came_from[nxt] = current
            heapq.heappush(frontier, (heuristic(nxt, problem.goal), nxt))

    path = reconstruct_path(came_from, problem.start, problem.goal)
    runtime_ms = (time.perf_counter() - start_time) * 1000

    return SearchResult(
        algorithm="Greedy Best-First Search",
        heuristic_name=heuristic_name,
        path=path,
        path_cost=g_score.get(problem.goal, math.inf),
        expanded_nodes=len(expansion_order),
        runtime_ms=runtime_ms,
        expansion_order=expansion_order,
        tracked_scores=tracked_scores,
    )


def greedy_beam_best_first(
    problem: GridProblem,
    heuristic: Heuristic,
    heuristic_name: str,
    beam_width: int = 3,
) -> SearchResult:
    start_time = time.perf_counter()

    frontier: List[Tuple[float, Point]] = []
    heapq.heappush(frontier, (heuristic(problem.start, problem.goal), problem.start))

    came_from: Dict[Point, Point] = {}
    g_score: Dict[Point, float] = {problem.start: 0.0}
    visited = set()
    expansion_order: List[Point] = []
    tracked_scores: List[Dict[str, float]] = []

    while frontier:
        beam: List[Point] = []
        while frontier and len(beam) < beam_width:
            _, current = heapq.heappop(frontier)
            if current in visited:
                continue
            visited.add(current)
            beam.append(current)

        if not beam:
            continue

        candidates: List[Tuple[float, Point]] = []
        for current in beam:
            expansion_order.append(current)
            h = heuristic(current, problem.goal)
            tracked_scores.append(
                {
                    "x": current[0],
                    "y": current[1],
                    "g": g_score[current],
                    "h": h,
                    "f": h,
                }
            )

            if current == problem.goal:
                frontier = []
                break

            for nxt in problem.neighbors(current):
                if nxt in visited:
                    continue
                step = problem.move_cost(nxt)
                new_cost = g_score[current] + step
                if nxt not in g_score or new_cost < g_score[nxt]:
                    g_score[nxt] = new_cost
                    came_from[nxt] = current
                candidates.append((heuristic(nxt, problem.goal), nxt))

        if not candidates:
            continue

        candidates.sort(key=lambda item: item[0])
        for score, node in candidates[:beam_width]:
            heapq.heappush(frontier, (score, node))

    path = reconstruct_path(came_from, problem.start, problem.goal)
    runtime_ms = (time.perf_counter() - start_time) * 1000

    return SearchResult(
        algorithm=f"Greedy Beam Best-First (k={beam_width})",
        heuristic_name=heuristic_name,
        path=path,
        path_cost=g_score.get(problem.goal, math.inf),
        expanded_nodes=len(expansion_order),
        runtime_ms=runtime_ms,
        expansion_order=expansion_order,
        tracked_scores=tracked_scores,
    )


def run_with_memory_tracking(search_fn: Callable[..., SearchResult], *args, **kwargs) -> SearchResult:
    tracemalloc.start()
    try:
        result = search_fn(*args, **kwargs)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    result.peak_memory_kb = peak_bytes / 1024.0
    return result


def astar(problem: GridProblem, heuristic: Heuristic, heuristic_name: str, weight: float = 1.0) -> SearchResult:
    start_time = time.perf_counter()

    frontier: List[Tuple[float, Point]] = []
    heapq.heappush(frontier, (0.0, problem.start))

    came_from: Dict[Point, Point] = {}
    g_score: Dict[Point, float] = {problem.start: 0.0}
    expansion_order: List[Point] = []
    tracked_scores: List[Dict[str, float]] = []

    closed = set()

    while frontier:
        _, current = heapq.heappop(frontier)
        if current in closed:
            continue
        closed.add(current)

        expansion_order.append(current)
        h = heuristic(current, problem.goal)
        tracked_scores.append({"x": current[0], "y": current[1], "g": g_score[current], "h": h, "f": g_score[current] + weight * h})

        if current == problem.goal:
            break

        for nxt in problem.neighbors(current):
            new_cost = g_score[current] + problem.move_cost(nxt)
            if nxt not in g_score or new_cost < g_score[nxt]:
                g_score[nxt] = new_cost
                came_from[nxt] = current
                score = new_cost + weight * heuristic(nxt, problem.goal)
                heapq.heappush(frontier, (score, nxt))

    path = reconstruct_path(came_from, problem.start, problem.goal)
    runtime_ms = (time.perf_counter() - start_time) * 1000

    variant = "A*" if weight == 1.0 else f"Weighted A* (w={weight})"
    return SearchResult(
        algorithm=variant,
        heuristic_name=heuristic_name,
        path=path,
        path_cost=g_score.get(problem.goal, math.inf),
        expanded_nodes=len(expansion_order),
        runtime_ms=runtime_ms,
        expansion_order=expansion_order,
        tracked_scores=tracked_scores,
    )


def shortest_cost_to_goal(problem: GridProblem) -> Dict[Point, float]:
    """Computes exact shortest cost from each node to goal using reverse Dijkstra."""
    dist: Dict[Point, float] = {problem.goal: 0.0}
    pq: List[Tuple[float, Point]] = [(0.0, problem.goal)]

    while pq:
        curr_dist, node = heapq.heappop(pq)
        if curr_dist > dist[node]:
            continue

        for prev in problem.neighbors(node):
            candidate = curr_dist + problem.move_cost(node)
            if prev not in dist or candidate < dist[prev]:
                dist[prev] = candidate
                heapq.heappush(pq, (candidate, prev))

    return dist


def analyze_heuristic(problem: GridProblem, heuristic: Heuristic, heuristic_name: str) -> Dict[str, object]:
    exact = shortest_cost_to_goal(problem)
    nodes = [
        (x, y)
        for x in range(problem.width)
        for y in range(problem.height)
        if problem.passable((x, y)) and (x, y) in exact
    ]

    admissible_violations = 0
    consistent_violations = 0

    for node in nodes:
        if heuristic(node, problem.goal) > exact[node] + 1e-9:
            admissible_violations += 1

        for neighbor in problem.neighbors(node):
            if neighbor not in exact:
                continue
            lhs = heuristic(node, problem.goal)
            rhs = problem.move_cost(neighbor) + heuristic(neighbor, problem.goal)
            if lhs > rhs + 1e-9:
                consistent_violations += 1

    return {
        "heuristic": heuristic_name,
        "checked_nodes": len(nodes),
        "admissible": admissible_violations == 0,
        "consistency": consistent_violations == 0,
        "admissibility_violations": admissible_violations,
        "consistency_violations": consistent_violations,
    }


def plot_grid_result(problem: GridProblem, result: SearchResult, output_file: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    for x in range(problem.width):
        for y in range(problem.height):
            node = (x, y)
            if node in problem.blocked:
                color = "black"
            elif node in problem.terrain_cost and problem.terrain_cost[node] > 1.0:
                color = "#f4a261"
            else:
                color = "#e9ecef"
            ax.add_patch(plt.Rectangle((x, y), 1, 1, facecolor=color, edgecolor="#adb5bd", linewidth=0.5))

    for i, node in enumerate(result.expansion_order[:120]):
        ax.text(node[0] + 0.05, node[1] + 0.72, str(i + 1), fontsize=6, color="#495057")

    if result.path:
        xs = [p[0] + 0.5 for p in result.path]
        ys = [p[1] + 0.5 for p in result.path]
        ax.plot(xs, ys, color="#2a9d8f", linewidth=2.5, label="Path")

    ax.scatter(problem.start[0] + 0.5, problem.start[1] + 0.5, c="#1d3557", s=100, marker="o", label="Start")
    ax.scatter(problem.goal[0] + 0.5, problem.goal[1] + 0.5, c="#e63946", s=100, marker="*", label="Goal")

    ax.set_xlim(0, problem.width)
    ax.set_ylim(0, problem.height)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title(f"{result.algorithm} | h(n)={result.heuristic_name}")
    ax.legend(loc="upper right")
    ax.set_xticks(range(problem.width + 1))
    ax.set_yticks(range(problem.height + 1))
    ax.grid(False)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=180)
    plt.close(fig)


def run_heuristic_search_experiments(output_dir: Path) -> Dict[str, object]:
    blocked = {
        (3, 0), (3, 1), (3, 2), (3, 3),
        (7, 2), (7, 3), (7, 4), (7, 5),
        (10, 0), (10, 1), (10, 2),
        (1, 7), (2, 7), (3, 7), (4, 7),
        (8, 8), (9, 8), (10, 8),
    }
    terrain_cost = {
        (5, 1): 3, (5, 2): 3, (5, 3): 3,
        (6, 6): 4, (7, 6): 4, (8, 6): 4,
        (11, 5): 5, (11, 6): 5,
    }

    problem = GridProblem(
        width=14,
        height=10,
        start=(0, 0),
        goal=(13, 9),
        blocked=blocked,
        terrain_cost=terrain_cost,
    )

    heuristics = {
        "manhattan": manhattan,
        "euclidean": euclidean,
    }

    results: List[SearchResult] = []
    for h_name, h_fn in heuristics.items():
        results.append(run_with_memory_tracking(greedy_best_first, problem, h_fn, h_name))
        results.append(run_with_memory_tracking(greedy_beam_best_first, problem, h_fn, h_name, beam_width=3))
        results.append(run_with_memory_tracking(astar, problem, h_fn, h_name, weight=1.0))
        results.append(run_with_memory_tracking(astar, problem, h_fn, h_name, weight=1.4))

    checks = [analyze_heuristic(problem, h_fn, h_name) for h_name, h_fn in heuristics.items()]

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "heuristic_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["algorithm", "heuristic", "path_length", "path_cost", "expanded_nodes", "runtime_ms", "peak_memory_kb"],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "algorithm": r.algorithm,
                    "heuristic": r.heuristic_name,
                    "path_length": len(r.path),
                    "path_cost": f"{r.path_cost:.3f}",
                    "expanded_nodes": r.expanded_nodes,
                    "runtime_ms": f"{r.runtime_ms:.3f}",
                    "peak_memory_kb": f"{r.peak_memory_kb:.2f}",
                }
            )

    path_summary = {
        f"{r.algorithm}__{r.heuristic_name}": {
            "path": r.path,
            "expansion_order": r.expansion_order,
            "tracked_scores_sample": r.tracked_scores[:25],
        }
        for r in results
    }
    with (output_dir / "heuristic_paths_and_expansions.json").open("w", encoding="utf-8") as f:
        json.dump(path_summary, f, indent=2)

    with (output_dir / "heuristic_validity_checks.json").open("w", encoding="utf-8") as f:
        json.dump(checks, f, indent=2)

    for r in results:
        safe_name = f"{r.algorithm.replace('*', 'star').replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')}_{r.heuristic_name}.png"
        plot_grid_result(problem, r, output_dir / safe_name)

    return {
        "scenario": {
            "grid_size": [problem.width, problem.height],
            "start": problem.start,
            "goal": problem.goal,
            "blocked_count": len(problem.blocked),
        },
        "results_count": len(results),
        "validity_checks": checks,
    }
