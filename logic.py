"""Battlesnake logic.py — food-planner + anti-trap strategy.

Drop-in replacement for Flask baseline.
Required public functions:
- get_info()
- choose_move(game_state)

Main strategy:
1. Never choose an instantly-dead move.
2. If hungry / small, choose a FOOD TARGET using BFS path planning.
3. Before committing to food, check that the route does not end in a trap:
   - enough room after reaching food;
   - preferably can still reach own tail after the route;
   - avoid one-way pockets unless panic-hungry.
4. If not eating, prefer tail-chasing / space-control, which is safer for long snakes.
5. Use head-to-head danger and enemy race-to-food checks.
6. Detect pressure near walls/enemy bodies and prefer escaping into open space.
7. If safely stronger, hunt by cutting enemy space / controlling gates, not by reckless chasing.
8. In dominate/food_denial modes, actively starve and box smaller opponents.

This file uses only stdlib and is defensive: if anything fails, choose_move returns
some legal-ish move instead of crashing the server.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

Point = Tuple[int, int]

DIRECTIONS: Dict[str, Point] = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

MAX_HEALTH = 100
PANIC_HEALTH = 25
HUNGRY_HEALTH = 60
GROW_UNTIL_LENGTH = 8
LONG_SNAKE_LENGTH = 12
DOMINATE_LENGTH_ADVANTAGE = 2
FOOD_DENIAL_ENEMY_HEALTH = 35

DEATH_SCORE = -1_000_000_000.0
BIG_PENALTY = 100_000.0

# Target memory: helps the snake not oscillate between food pieces.
# Battlesnake usually runs one Python process per deployed bot, so this is OK.
TARGET_FOOD_BY_GAME: Dict[str, Point] = {}


@dataclass(frozen=True)
class Candidate:
    move: str
    point: Point


@dataclass(frozen=True)
class ScoredMove:
    move: str
    score: float
    reasons: Dict[str, float]


@dataclass(frozen=True)
class FoodPlan:
    food: Point
    path: List[Point]  # includes start and food
    score: float
    safe_after_food: bool
    reaches_tail_after_food: bool
    room_after_food: int
    first_move: str


# ---------------------------------------------------------------------------
# Public API expected by backend.py
# ---------------------------------------------------------------------------


def get_info() -> Dict[str, str]:
    return {
        "apiversion": "1",
        "author": "hackathon",
        "color": "#E80978",
        "head": "do-sammy",
        "tail": "weight",
        "version": "4.0.0-killer-dominate",
    }



def choose_move(game_state: Dict) -> str:
    """Main decision function called from /move.

    Important tournament guard:
    even if the strategic layer returns a bad direction or raises an exception,
    the final answer is validated against the CURRENT board before returning.
    This prevents accidental moves outside the board, including the old unsafe
    fallback `return "up"` case when something went wrong.
    """
    try:
        move = choose_move_safe(game_state)
    except Exception:
        move = None
    return _guarded_final_move(game_state, move)


# ---------------------------------------------------------------------------
# Top-level strategy
# ---------------------------------------------------------------------------


def choose_move_safe(game_state: Dict) -> str:
    board = game_state["board"]
    you = game_state["you"]
    width, height = int(board["width"]), int(board["height"])
    head = _head(you)
    mode = _mode(board, you)

    candidates = _safe_immediate_candidates(board, you)
    if not candidates:
        return fallback_move(game_state)

    # 1) Food planner has priority when hungry/small.
    # Also allow food in control mode if it is very close and very safe.
    food_plan: Optional[FoodPlan] = None
    if mode in {"panic_food", "food", "grow"}:
        food_plan = choose_food_plan(board, you, mode, candidates)
    elif mode in {"control", "hunt", "dominate"}:
        food_plan = choose_food_plan(board, you, "opportunistic_food", candidates)

    if food_plan is not None:
        return food_plan.first_move

    # 2) If long and healthy, prefer following the tail / staying cyclic.
    # This is the main anti-trap behavior: a long snake survives by keeping a
    # route to its own tail instead of greedily entering pockets.
    if mode in {"tail_chase", "control", "survival"}:
        tail_move = choose_tail_chase_move(board, you, candidates)
        if tail_move is not None:
            return tail_move

    # 3) General space-control fallback.
    scores = score_space_moves(board, you, candidates, mode)
    if scores:
        return max(scores, key=lambda x: x.score).move

    # 4) Last resort: any safe candidate.
    return candidates[0].move



def _mode(board: Dict, you: Dict) -> str:
    health = int(you["health"])
    length = int(you["length"])
    width, height = int(board["width"]), int(board["height"])
    head = _head(you)

    if health <= PANIC_HEALTH:
        return "panic_food"

    blocked = _blocked_for_planning(board, you, assume_our_next=None)
    blocked.discard(head)
    current_room = _flood_fill(head, blocked, width, height, limit=width * height)
    pressure = _pressure_score(board, you, head)

    # If we are already being squeezed, escape pressure before chasing food.
    # Exception: panic hunger, handled above.
    if pressure >= 5 or (pressure >= 3 and current_room < max(length * 2, length + 10)):
        return "escape_pressure"

    if length < GROW_UNTIL_LENGTH:
        return "grow"

    # If already cramped, do not think about fancy attack/food. Survive.
    if current_room < max(length * 2, length + 8):
        return "survival"

    # If an opponent is hungry, blocking its food can be stronger than eating ourselves.
    # Do this only when our own health is not urgent.
    if health > 35 and _can_food_denial(board, you):
        return "food_denial"

    # If we have a length advantage, use it as a weapon: control center, cut exits,
    # pressure smaller heads. This intentionally comes before normal food.
    if health > 40 and _can_dominate(board, you):
        return "dominate"

    if health <= HUNGRY_HEALTH:
        return "food"

    # Hunt only when not hungry, not pressured, and there is a smaller enemy to cut off.
    if _can_hunt(board, you):
        return "hunt"

    # Long healthy snakes should stop overeating and maintain a tail route.
    if length >= LONG_SNAKE_LENGTH and health > HUNGRY_HEALTH:
        return "tail_chase"

    return "control"


# ---------------------------------------------------------------------------
# Food planner
# ---------------------------------------------------------------------------


def choose_food_plan(
    board: Dict,
    you: Dict,
    mode: str,
    candidates: Sequence[Candidate],
) -> Optional[FoodPlan]:
    foods = list(_food_set(board))
    if not foods:
        _clear_target(board)
        return None

    width, height = int(board["width"]), int(board["height"])
    head = _head(you)
    health = int(you["health"])
    length = int(you["length"])
    game_id = _game_id(board)

    blocked = _blocked_for_planning(board, you, assume_our_next=None)
    # Let BFS start from our head.
    blocked.discard(head)

    candidate_moves = {c.point: c.move for c in candidates}

    # Try existing target first, but only if it still exists and is still sane.
    old_target = TARGET_FOOD_BY_GAME.get(game_id)
    old_plan: Optional[FoodPlan] = None
    if old_target in foods:
        path = _bfs_path(head, old_target, blocked, width, height)
        if path and len(path) >= 2 and path[1] in candidate_moves:
            old_plan = _build_food_plan(board, you, path, candidate_moves[path[1]], mode)
            if old_plan and _plan_is_acceptable(old_plan, board, you, mode, sticky=True):
                return old_plan

    plans: List[FoodPlan] = []
    for food in foods:
        path = _bfs_path(head, food, blocked, width, height)
        if not path or len(path) < 2:
            continue
        first_step = path[1]
        if first_step not in candidate_moves:
            continue

        # Can we reach food before starving? Distance in turns is len(path)-1.
        dist = len(path) - 1
        if dist >= health:
            continue

        plan = _build_food_plan(board, you, path, candidate_moves[first_step], mode)
        if plan is None:
            continue
        if _plan_is_acceptable(plan, board, you, mode, sticky=False):
            plans.append(plan)

    if not plans:
        _clear_target(board)
        return None

    best = max(plans, key=lambda p: p.score)

    # Opportunistic mode should only eat if food is clearly attractive.
    if mode == "opportunistic_food" and best.score < 1_600:
        return None

    TARGET_FOOD_BY_GAME[game_id] = best.food
    return best



def _build_food_plan(
    board: Dict,
    you: Dict,
    path: List[Point],
    first_move: str,
    mode: str,
) -> Optional[FoodPlan]:
    width, height = int(board["width"]), int(board["height"])
    health = int(you["health"])
    length = int(you["length"])
    food = path[-1]
    dist = len(path) - 1

    # Race check: if equal/bigger enemy can arrive first or same turn, avoid it
    # unless we are in panic and have no luxury.
    race_penalty = _enemy_race_penalty(board, you, food, my_dist=dist)
    if race_penalty >= BIG_PENALTY and mode != "panic_food":
        return None

    sim_body, sim_blocked = _simulate_our_body_after_path(board, you, path)
    sim_head = path[-1]
    sim_tail = sim_body[-1] if sim_body else sim_head

    # Important anti-trap checks after reaching food.
    room_after_food = _flood_fill(sim_head, sim_blocked - {sim_head}, width, height, limit=width * height)
    reaches_tail = _reachable(sim_head, sim_tail, sim_blocked - {sim_head, sim_tail}, width, height)
    exits_after_food = _escape_routes(sim_head, sim_blocked - {sim_head}, width, height)

    # New length after eating once at the target.
    expected_len = length + 1

    safe_after_food = True
    if room_after_food < expected_len:
        safe_after_food = False
    if exits_after_food == 0:
        safe_after_food = False
    # For long snakes, tail reachability is more important than raw room size.
    if expected_len >= LONG_SNAKE_LENGTH and not reaches_tail and room_after_food < expected_len * 2:
        safe_after_food = False

    # Score food as a target.
    score = 0.0

    # Health urgency.
    if mode == "panic_food":
        score += 12_000.0 / max(1, dist)
        score += 180.0 * (PANIC_HEALTH - min(health, PANIC_HEALTH))
    elif mode == "food":
        score += 6_000.0 / max(1, dist)
        score += 70.0 * (HUNGRY_HEALTH - min(health, HUNGRY_HEALTH))
    elif mode == "grow":
        score += 4_500.0 / max(1, dist)
        score += 350.0 * max(0, GROW_UNTIL_LENGTH - length)
    else:  # opportunistic_food
        score += 1_500.0 / max(1, dist)

    # Prefer closer food, but not at any price.
    score -= 80.0 * dist

    # Anti-trap value.
    score += min(room_after_food, expected_len * 4) * 35.0
    score += 1_600.0 if reaches_tail else -2_800.0
    score += 650.0 * exits_after_food

    if safe_after_food:
        score += 1_500.0
    else:
        # Panic may accept unsafe food if starvation is imminent; others should not.
        score -= 7_500.0 if mode != "panic_food" else 1_500.0

    score -= race_penalty

    # Food on the edge/corner is more likely to become a trap for long snakes.
    wall_dist = min(food[0], width - 1 - food[0], food[1], height - 1 - food[1])
    if wall_dist == 0 and length >= GROW_UNTIL_LENGTH:
        score -= 900.0
    elif wall_dist == 1 and length >= LONG_SNAKE_LENGTH:
        score -= 500.0

    # Do not let food pull a healthy snake into pressure or a corridor pocket.
    pressure_at_food = _pressure_score(board, you, food)
    chamber = _chamber_info(sim_head, sim_blocked - {sim_head}, width, height)
    if mode != "panic_food":
        score -= 550.0 * pressure_at_food
        if chamber["junctions"] == 0 and chamber["size"] < expected_len * 3:
            score -= 3_500.0
        if pressure_at_food >= 4 and not reaches_tail:
            score -= 5_000.0

    return FoodPlan(
        food=food,
        path=path,
        score=score,
        safe_after_food=safe_after_food,
        reaches_tail_after_food=reaches_tail,
        room_after_food=room_after_food,
        first_move=first_move,
    )



def _plan_is_acceptable(plan: FoodPlan, board: Dict, you: Dict, mode: str, sticky: bool) -> bool:
    health = int(you["health"])
    length = int(you["length"])
    dist = len(plan.path) - 1

    # Immediate starvation: accept almost anything that is not instant death.
    if mode == "panic_food":
        if dist >= health:
            return False
        return plan.room_after_food >= max(3, min(length // 2, 8))

    # Non-panic: do not walk into known food traps.
    if not plan.safe_after_food:
        return False

    # If long, insist on reaching tail or having a lot of room after eating.
    if length >= LONG_SNAKE_LENGTH:
        if not plan.reaches_tail_after_food and plan.room_after_food < length * 2:
            return False

    # Sticky target can be slightly lower score to avoid oscillation.
    min_score = 300.0 if sticky else 600.0
    if mode == "opportunistic_food":
        min_score = 1_600.0
    return plan.score >= min_score


# ---------------------------------------------------------------------------
# Tail chase and space-control
# ---------------------------------------------------------------------------


def choose_tail_chase_move(board: Dict, you: Dict, candidates: Sequence[Candidate]) -> Optional[str]:
    """Try to move along a safe path to our own tail.

    This is extremely useful after the snake becomes long: following the tail
    naturally prevents self-enclosure because the target keeps moving away.
    """
    width, height = int(board["width"]), int(board["height"])
    head = _head(you)
    tail = _tail(you)
    length = int(you["length"])

    blocked = _blocked_for_planning(board, you, assume_our_next=None)
    blocked.discard(head)
    blocked.discard(tail)
    path = _bfs_path(head, tail, blocked, width, height)
    if not path or len(path) < 2:
        return None

    first = path[1]
    candidate_by_point = {c.point: c.move for c in candidates}
    if first not in candidate_by_point:
        return None

    # Do not tail-chase into an obviously tiny room.
    move = candidate_by_point[first]
    candidate_blocked = _blocked_for_planning(board, you, assume_our_next=first)
    room = _flood_fill(first, candidate_blocked - {first}, width, height, limit=width * height)
    exits = _escape_routes(first, candidate_blocked - {first}, width, height)
    danger = _enemy_head_next_cells(board, you, min_len=length)

    if first in danger:
        return None
    if room < max(length, 8):
        return None
    if exits == 0:
        return None

    return move



def score_space_moves(board: Dict, you: Dict, candidates: Sequence[Candidate], mode: str) -> List[ScoredMove]:
    width, height = int(board["width"]), int(board["height"])
    length = int(you["length"])
    health = int(you["health"])
    head = _head(you)
    tail = _tail(you)
    foods = _food_set(board)
    hazards = _hazard_set(board)
    hazard_damage = _hazard_damage(board)
    danger_equal_bigger = _enemy_head_next_cells(board, you, min_len=length)
    smaller_head_next = _enemy_head_next_cells(board, you, max_len=length - 1)

    scored: List[ScoredMove] = []

    for cand in candidates:
        p = cand.point
        reasons: Dict[str, float] = {}
        score = 0.0

        blocked = _blocked_for_planning(board, you, assume_our_next=p)
        blocked.discard(p)

        room = _flood_fill(p, blocked, width, height, limit=width * height)
        exits = _escape_routes(p, blocked, width, height)
        next_safe = _next_safe_move_count(board, you, p)
        reaches_tail = _reachable(p, tail, blocked - {tail}, width, height)
        vor = _voronoi_control(board, you, p, blocked, width, height)
        pressure = _pressure_score(board, you, p)
        chamber = _chamber_info(p, blocked, width, height)
        enemy_cut = _enemy_space_reduction(board, you, p)
        attack = _attack_features_after_move(board, you, p)
        dist_small_head = _distance_to_nearest_smaller_head(board, you, p)

        reasons["pressure"] = float(pressure)
        reasons["chamber_size"] = float(chamber["size"])
        reasons["chamber_junctions"] = float(chamber["junctions"])
        reasons["enemy_cut"] = float(enemy_cut)
        reasons["enemy_escape_reduction"] = float(attack["escape_reduction"])
        reasons["enemy_pressure_gain"] = float(attack["pressure_gain"])
        reasons["food_denial"] = float(attack["food_denial"])
        reasons["center_cut"] = float(attack["center_cut"])
        reasons["head_control"] = float(attack["head_control"])

        reasons["room"] = float(room)
        reasons["exits"] = float(exits)
        reasons["next_safe"] = float(next_safe)
        reasons["reaches_tail"] = 1.0 if reaches_tail else 0.0
        reasons["voronoi"] = float(vor)

        # Hard-ish danger.
        if p in danger_equal_bigger:
            score -= 120_000.0
            reasons["h2h_risk"] = 1.0
        else:
            reasons["h2h_risk"] = 0.0

        # Anti-trap: room relative to length.
        score += 40.0 * min(room, length * 5)
        if room < length:
            score -= 50_000.0 * (1.0 - room / max(1, length))
        elif room < length * 2:
            score -= 4_000.0
        elif room >= length * 3:
            score += 1_500.0

        # Exits and one-ply safety.
        score += 900.0 * exits
        score += 1_000.0 * next_safe
        if exits == 0 or next_safe == 0:
            score -= 40_000.0
        elif exits == 1:
            # One-exit pockets are usually traps for long snakes.
            score -= 2_500.0 if length < LONG_SNAKE_LENGTH else 7_000.0

        # Tail reachability is the strongest long-snake survival signal.
        if reaches_tail:
            score += 4_500.0 if length < LONG_SNAKE_LENGTH else 9_000.0
        else:
            score -= 2_500.0 if length < LONG_SNAKE_LENGTH else 8_000.0

        # Territory.
        score += 7.0 * vor

        # Pressure / anti-squeeze: do not allow enemies to pin us against walls or bodies.
        score -= 950.0 * pressure
        if pressure >= 5 and room < length * 3:
            score -= 8_000.0

        # Chamber structure: a large flood-fill can still be bad if it is a long corridor
        # with no junctions. Prefer open rooms; avoid one-gate pockets.
        if chamber["size"] < length:
            score -= 30_000.0
        if chamber["junctions"] == 0 and chamber["size"] < length * 3:
            score -= 4_500.0
        elif chamber["junctions"] >= 2:
            score += min(chamber["junctions"], 6) * 700.0

        # Safe aggression: use length advantage to cut exits, deny food and control heads.
        # The safety gate prevents reckless suicide-attacks.
        safe_to_pressure = room >= length * 2 and exits >= 2 and next_safe >= 2
        if safe_to_pressure:
            score += 140.0 * enemy_cut
            score += 900.0 * attack["escape_reduction"]
            score += 650.0 * attack["pressure_gain"]
            score += 800.0 * attack["food_denial"]
            score += 550.0 * attack["center_cut"]
            score += 700.0 * attack["head_control"]

        # Direct head control: adjacent cells to smaller heads are valuable when we are longer.
        if p in smaller_head_next and safe_to_pressure:
            score += 5_500.0 if mode in {"hunt", "dominate", "food_denial"} else 3_000.0
            reasons["attack"] = 1.0
        else:
            reasons["attack"] = 0.0

        # In hunt/dominate modes, staying close to a smaller head can be useful, but only
        # as positional pressure. Do not reward it if we are boxed in.
        if mode in {"hunt", "dominate", "food_denial"} and safe_to_pressure:
            if dist_small_head <= 3:
                score += 1_800.0 / max(1, dist_small_head)
            score += 220.0 * enemy_cut
            score += 1_200.0 * attack["escape_reduction"]
            score += 1_000.0 * attack["food_denial"]

        # Avoid unnecessary eating when long and healthy.
        if p in foods:
            if health <= HUNGRY_HEALTH:
                score += 1_500.0
            elif length >= LONG_SNAKE_LENGTH:
                score -= 1_800.0
            else:
                score += 200.0

        # Hazards.
        if p in hazards:
            score -= 1_200.0 + 60.0 * hazard_damage

        # Small center/wall preference. Never dominates survival.
        center_dist = _dist_to_center(p, width, height)
        score -= 8.0 * center_dist
        wall_dist = min(p[0], width - 1 - p[0], p[1], height - 1 - p[1])
        if wall_dist == 0 and length >= GROW_UNTIL_LENGTH:
            score -= 700.0
        else:
            score += 35.0 * wall_dist

        # Mode shaping.
        if mode == "survival":
            score += 35.0 * room
            score += 4_000.0 if reaches_tail else -4_000.0
        elif mode == "tail_chase":
            score += 6_000.0 if reaches_tail else -5_000.0
            # Don't keep expanding indefinitely.
            if p in foods and health > HUNGRY_HEALTH:
                score -= 2_000.0
        elif mode == "escape_pressure":
            # Primary goal: get away from wall/body squeeze into an open multi-exit chamber.
            score += 70.0 * room
            score += 3_500.0 * exits
            score -= 1_600.0 * pressure
            score += 4_000.0 if reaches_tail else -2_500.0
            # Prefer moving toward center while escaping.
            score -= 40.0 * center_dist
        elif mode == "food_denial":
            # Starve weak opponents: block or steal the food they need, while staying safe.
            score += 18.0 * vor
            score += 280.0 * enemy_cut
            score += 2_000.0 * attack["food_denial"]
            score += 1_200.0 * attack["escape_reduction"]
            score -= 1_100.0 * pressure
            if exits < 2 or next_safe < 2:
                score -= 9_000.0
        elif mode == "dominate":
            # Dominance: hold center/open ground and squeeze smaller snakes.
            score += 18.0 * vor
            score += 320.0 * enemy_cut
            score += 1_400.0 * attack["escape_reduction"]
            score += 1_100.0 * attack["pressure_gain"]
            score += 900.0 * attack["center_cut"]
            score += 900.0 * attack["head_control"]
            score -= 1_000.0 * pressure
            # A dominant snake should not be greedy for edge food.
            if p in foods and health > HUNGRY_HEALTH:
                score -= 1_600.0
            if exits < 2 or next_safe < 2:
                score -= 10_000.0
            if not reaches_tail and room < length * 3:
                score -= 8_000.0
        elif mode == "hunt":
            # Hunt by space cutting, not suicide chasing.
            score += 12.0 * vor
            score += 240.0 * enemy_cut
            score += 900.0 * attack["escape_reduction"]
            score -= 1_200.0 * pressure
            if exits < 2 or not reaches_tail:
                score -= 7_000.0
            # Avoid eating during a good hunt unless hungry; growth can ruin the cut.
            if p in foods and health > HUNGRY_HEALTH:
                score -= 1_200.0
        elif mode == "control":
            score += 3.0 * vor
            score += 8.0 * room

        scored.append(ScoredMove(cand.move, score, reasons))

    return scored


# ---------------------------------------------------------------------------
# Candidate generation and immediate safety
# ---------------------------------------------------------------------------


def _safe_immediate_candidates(board: Dict, you: Dict) -> List[Candidate]:
    width, height = int(board["width"]), int(board["height"])
    head = _head(you)
    length = int(you["length"])
    health = int(you["health"])
    foods = _food_set(board)
    hazards = _hazard_set(board)
    hazard_damage = _hazard_damage(board)
    danger_equal_bigger = _enemy_head_next_cells(board, you, min_len=length)

    result: List[Candidate] = []
    soft_result: List[Candidate] = []

    for move, delta in DIRECTIONS.items():
        p = _add(head, delta)
        if not _in_bounds(p, width, height):
            continue

        blocked = _blocked_for_immediate_move(board, you, p)
        if p in blocked:
            continue

        # Hazard can kill before the next turn. Eating may restore health, so allow
        # food-in-hazard unless damage makes it impossible in your ruleset.
        if p in hazards and p not in foods and health - 1 - hazard_damage <= 0:
            continue

        cand = Candidate(move, p)
        if p in danger_equal_bigger:
            # Keep as soft fallback: sometimes every move is h2h-dangerous.
            soft_result.append(cand)
        else:
            result.append(cand)

    return result or soft_result



def fallback_move(game_state: Dict) -> str:
    """Last-resort move selection. Never intentionally returns an off-board move."""
    return _guarded_final_move(game_state, None)


def _guarded_final_move(game_state: Dict, proposed_move: Optional[str]) -> str:
    """Validate the final move against the current board.

    Priority:
    1. keep proposed_move if it is inside board and not an immediate body collision;
    2. use one of the strategic safe candidates;
    3. use any in-bounds non-occupied move;
    4. use any in-bounds move even if it hits a body, because body collision is still
       preferable to walking into a wall when the position is already lost;
    5. return "up" only if the input state itself is malformed.
    """
    try:
        board = game_state["board"]
        you = game_state["you"]
        width, height = int(board["width"]), int(board["height"])
        head = _head(you)

        def point_for(move: str) -> Point:
            return _add(head, DIRECTIONS[move])

        # Immediate collision set with smart own-tail handling for each candidate.
        if proposed_move in DIRECTIONS:
            p = point_for(proposed_move)
            if _in_bounds(p, width, height):
                try:
                    blocked = _blocked_for_immediate_move(board, you, p)
                    if p not in blocked:
                        return proposed_move
                except Exception:
                    # If collision logic itself fails, at least do not walk into a wall.
                    return proposed_move

        # Recompute official immediate-safe candidates.
        try:
            candidates = _safe_immediate_candidates(board, you)
            if candidates:
                return candidates[0].move
        except Exception:
            pass

        # Any in-bounds non-occupied move.
        occupied = _occupied_cells(board.get("snakes", []))
        for move, delta in DIRECTIONS.items():
            p = _add(head, delta)
            if _in_bounds(p, width, height) and p not in occupied:
                return move

        # If every in-bounds move collides with a body, still choose in-bounds.
        for move, delta in DIRECTIONS.items():
            p = _add(head, delta)
            if _in_bounds(p, width, height):
                return move
    except Exception:
        pass
    return "up"



# ---------------------------------------------------------------------------
# Pressure, chambers, and hunting
# ---------------------------------------------------------------------------

def _can_hunt(board: Dict, you: Dict) -> bool:
    """We hunt only if there is a smaller enemy and we are not cramped."""
    my_len = int(you.get("length", len(you.get("body", []))))
    if int(you.get("health", 0)) <= HUNGRY_HEALTH:
        return False
    width, height = int(board["width"]), int(board["height"])
    head = _head(you)
    blocked = _blocked_for_planning(board, you, assume_our_next=None)
    blocked.discard(head)
    room = _flood_fill(head, blocked, width, height, width * height)
    if room < max(my_len * 2, my_len + 10):
        return False
    for enemy in board.get("snakes", []):
        if enemy.get("id") == you.get("id"):
            continue
        e_len = int(enemy.get("length", len(enemy.get("body", []))))
        if e_len < my_len:
            return True
    return False



def _can_dominate(board: Dict, you: Dict) -> bool:
    """True when we have enough length advantage to play as a territory controller."""
    my_len = int(you.get("length", len(you.get("body", []))))
    enemies = [s for s in board.get("snakes", []) if s.get("id") != you.get("id")]
    if not enemies:
        return False
    max_enemy = max(int(e.get("length", len(e.get("body", [])))) for e in enemies)
    if my_len < max_enemy + DOMINATE_LENGTH_ADVANTAGE:
        return False
    width, height = int(board["width"]), int(board["height"])
    head = _head(you)
    blocked = _blocked_for_planning(board, you, assume_our_next=None)
    blocked.discard(head)
    room = _flood_fill(head, blocked, width, height, width * height)
    return room >= max(my_len * 2, my_len + 10)


def _can_food_denial(board: Dict, you: Dict) -> bool:
    """True if there is a hungry smaller enemy whose food path we may be able to block."""
    my_len = int(you.get("length", len(you.get("body", []))))
    foods = list(_food_set(board))
    if not foods:
        return False
    for enemy in board.get("snakes", []):
        if enemy.get("id") == you.get("id"):
            continue
        e_len = int(enemy.get("length", len(enemy.get("body", []))))
        e_health = int(enemy.get("health", 100))
        if e_len < my_len and e_health <= FOOD_DENIAL_ENEMY_HEALTH:
            return True
    return False


def _enemy_wall_pressure(board: Dict, cell: Point) -> float:
    """Wall/corner pressure for an enemy head. Higher means easier to trap."""
    width, height = int(board["width"]), int(board["height"])
    score = 0.0
    if cell[0] == 0 or cell[0] == width - 1:
        score += 1.5
    elif cell[0] == 1 or cell[0] == width - 2:
        score += 0.5
    if cell[1] == 0 or cell[1] == height - 1:
        score += 1.5
    elif cell[1] == 1 or cell[1] == height - 2:
        score += 0.5
    return score


def _enemy_safe_moves_count(board: Dict, enemy: Dict, blocked: Set[Point]) -> int:
    width, height = int(board["width"]), int(board["height"])
    h = _head(enemy)
    count = 0
    for delta in DIRECTIONS.values():
        p = _add(h, delta)
        if _in_bounds(p, width, height) and p not in blocked:
            count += 1
    return count


def _nearest_food_info(board: Dict, start: Point, blocked: Set[Point]) -> Optional[Tuple[Point, int]]:
    foods = _food_set(board)
    if not foods:
        return None
    width, height = int(board["width"]), int(board["height"])
    dist = _bfs_dist([start], blocked - {start}, width, height)
    reachable = [(f, dist[f]) for f in foods if f in dist]
    if not reachable:
        return None
    return min(reachable, key=lambda x: x[1])


def _attack_features_after_move(board: Dict, you: Dict, my_next: Point) -> Dict[str, float]:
    """Aggressive features for killing/boxing smaller opponents.

    The features are deliberately simple and deterministic:
    - escape_reduction: how many legal exits smaller enemies lose after our move;
    - pressure_gain: added wall/body pressure on smaller enemies;
    - food_denial: whether our move blocks/steals food for a hungry smaller enemy;
    - center_cut: whether we stand between a wall-side enemy and the center;
    - head_control: safe positional control near smaller heads.
    """
    width, height = int(board["width"]), int(board["height"])
    my_len = int(you.get("length", len(you.get("body", []))))
    center = ((width - 1) / 2.0, (height - 1) / 2.0)

    base_blocked = _blocked_for_planning(board, you, assume_our_next=None)
    after_blocked = _blocked_for_planning(board, you, assume_our_next=my_next)
    after_blocked.add(my_next)

    out = {
        "escape_reduction": 0.0,
        "pressure_gain": 0.0,
        "food_denial": 0.0,
        "center_cut": 0.0,
        "head_control": 0.0,
    }

    for enemy in board.get("snakes", []):
        if enemy.get("id") == you.get("id"):
            continue
        e_len = int(enemy.get("length", len(enemy.get("body", []))))
        if e_len >= my_len:
            continue

        e_head = _head(enemy)
        before_moves = _enemy_safe_moves_count(board, enemy, base_blocked - {e_head})
        after_moves = _enemy_safe_moves_count(board, enemy, after_blocked - {e_head})
        reduction = max(0, before_moves - after_moves)
        out["escape_reduction"] += reduction
        if after_moves <= 1:
            out["escape_reduction"] += 1.5
        if after_moves == 0:
            out["escape_reduction"] += 3.0

        # Our new head/body near enemy head adds squeeze pressure, especially near walls.
        d = _manhattan(my_next, e_head)
        if d == 1:
            out["head_control"] += 2.0
            out["pressure_gain"] += 1.0 + _enemy_wall_pressure(board, e_head)
        elif d == 2:
            out["head_control"] += 0.8
            out["pressure_gain"] += 0.35 * _enemy_wall_pressure(board, e_head)

        # Center cut: if enemy is wall-side, standing closer to center and near the enemy
        # blocks its escape route back to the open board.
        enemy_center_dist = abs(e_head[0] - center[0]) + abs(e_head[1] - center[1])
        our_center_dist = abs(my_next[0] - center[0]) + abs(my_next[1] - center[1])
        if _enemy_wall_pressure(board, e_head) >= 1.0 and our_center_dist < enemy_center_dist and d <= 3:
            out["center_cut"] += 1.0 + (3 - min(d, 3)) * 0.5

        # Food denial: hungry smaller enemy wants its nearest food. We can steal it, stand
        # adjacent to it, or step onto the shortest path region before it arrives.
        e_health = int(enemy.get("health", 100))
        if e_health <= FOOD_DENIAL_ENEMY_HEALTH:
            info = _nearest_food_info(board, e_head, base_blocked - {e_head})
            if info is not None:
                food, e_dist = info
                my_dist_to_food = _manhattan(my_next, food)
                if my_next == food:
                    out["food_denial"] += 4.0
                elif my_dist_to_food == 1 and e_dist <= max(4, e_health):
                    out["food_denial"] += 2.0
                elif my_dist_to_food < e_dist and d <= 4:
                    out["food_denial"] += 1.0

    return out


def _pressure_score(board: Dict, you: Dict, cell: Point) -> float:
    """How strongly this cell is squeezed by walls, enemy bodies and dangerous heads."""
    width, height = int(board["width"]), int(board["height"])
    my_len = int(you.get("length", len(you.get("body", []))))
    score = 0.0

    # Walls: corner is much worse than a single wall.
    if cell[0] == 0 or cell[0] == width - 1:
        score += 1.4
    elif cell[0] == 1 or cell[0] == width - 2:
        score += 0.45
    if cell[1] == 0 or cell[1] == height - 1:
        score += 1.4
    elif cell[1] == 1 or cell[1] == height - 2:
        score += 0.45

    for snake in board.get("snakes", []):
        if snake.get("id") == you.get("id"):
            continue
        enemy_len = int(snake.get("length", len(snake.get("body", []))))
        enemy_head = _head(snake)
        d_head = _manhattan(cell, enemy_head)

        # Big/equal heads squeeze us; small heads are attack opportunities, not pressure.
        if enemy_len >= my_len:
            if d_head == 1:
                score += 4.0
            elif d_head == 2:
                score += 1.4
        else:
            if d_head == 1:
                score -= 0.8

        # Enemy bodies next to us restrict escape lines.
        for p in _body(snake):
            if _manhattan(cell, p) == 1:
                score += 0.7

    return score


def _chamber_info(start: Point, blocked: Set[Point], width: int, height: int) -> Dict[str, int]:
    """Approximate local room structure behind a move.

    size: reachable free cells.
    junctions: cells with >=3 free neighbours. More junctions means less corridor-like.
    corridor_cells: cells with <=2 free neighbours. Many corridor cells means bottleneck/dead-end risk.
    gates: cells with exactly 2 free neighbours; rough proxy for narrow passage.
    """
    if not _in_bounds(start, width, height) or start in blocked:
        return {"size": 0, "junctions": 0, "corridor_cells": 0, "gates": 0}
    q: deque[Point] = deque([start])
    seen: Set[Point] = {start}
    junctions = 0
    corridor_cells = 0
    gates = 0
    while q:
        cur = q.popleft()
        free_neigh = 0
        for delta in DIRECTIONS.values():
            nxt = _add(cur, delta)
            if not _in_bounds(nxt, width, height) or nxt in blocked:
                continue
            free_neigh += 1
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
        if free_neigh >= 3:
            junctions += 1
        if free_neigh <= 2:
            corridor_cells += 1
        if free_neigh == 2:
            gates += 1
    return {"size": len(seen), "junctions": junctions, "corridor_cells": corridor_cells, "gates": gates}


def _enemy_space_reduction(board: Dict, you: Dict, my_next: Point) -> float:
    """Positive when our move cuts space from smaller enemies.

    This is the main hunt signal. It rewards occupying a gate / cutting line when
    it materially reduces a smaller enemy's flood-fill area while we stay safe.
    """
    width, height = int(board["width"]), int(board["height"])
    my_len = int(you.get("length", len(you.get("body", []))))
    base_blocked = _blocked_for_planning(board, you, assume_our_next=None)
    after_blocked = _blocked_for_planning(board, you, assume_our_next=my_next)
    after_blocked.add(my_next)
    gain = 0.0
    for enemy in board.get("snakes", []):
        if enemy.get("id") == you.get("id"):
            continue
        e_len = int(enemy.get("length", len(enemy.get("body", []))))
        if e_len >= my_len:
            continue
        e_head = _head(enemy)
        before = _flood_fill(e_head, base_blocked - {e_head}, width, height, width * height)
        after = _flood_fill(e_head, after_blocked - {e_head}, width, height, width * height)
        reduction = max(0, before - after)
        # More valuable if enemy becomes smaller than its body needs.
        if after < e_len:
            gain += reduction * 2.5 + 25
        else:
            gain += reduction
    return gain


def _distance_to_nearest_smaller_head(board: Dict, you: Dict, cell: Point) -> int:
    my_len = int(you.get("length", len(you.get("body", []))))
    best = 10_000
    for enemy in board.get("snakes", []):
        if enemy.get("id") == you.get("id"):
            continue
        e_len = int(enemy.get("length", len(enemy.get("body", []))))
        if e_len < my_len:
            best = min(best, _manhattan(cell, _head(enemy)))
    return best

# ---------------------------------------------------------------------------
# Trap / simulation helpers
# ---------------------------------------------------------------------------


def _simulate_our_body_after_path(board: Dict, you: Dict, path: Sequence[Point]) -> Tuple[List[Point], Set[Point]]:
    """Approximate our body after following a path to food.

    path includes current head and target. We assume the final target is food and
    therefore the body grows by one on the final step.
    Enemy bodies are treated mostly static/conservative.
    """
    foods = _food_set(board)
    body = list(_body(you))

    # Move through path[1:], growing only when stepping on food.
    for step in path[1:]:
        will_eat = step in foods
        body = [step] + body
        if not will_eat:
            body = body[:-1]

    blocked: Set[Point] = set(body)
    for snake in board.get("snakes", []):
        if snake.get("id") == you.get("id"):
            continue
        enemy_body = list(_body(snake))
        # For anti-trap room estimate, allow enemy tail to move if no adjacent food.
        if enemy_body and not _has_adjacent_food(_head(snake), foods):
            enemy_body = enemy_body[:-1]
        blocked.update(enemy_body)

    return body, blocked



def _next_safe_move_count(board: Dict, you: Dict, candidate: Point) -> int:
    width, height = int(board["width"]), int(board["height"])
    length = int(you["length"])
    blocked = _blocked_for_planning(board, you, assume_our_next=candidate)
    blocked.discard(candidate)
    danger = _enemy_head_next_cells(board, you, min_len=length)

    count = 0
    for delta in DIRECTIONS.values():
        p = _add(candidate, delta)
        if not _in_bounds(p, width, height):
            continue
        if p in blocked:
            continue
        if p in danger:
            continue
        count += 1
    return count


# ---------------------------------------------------------------------------
# Occupancy and enemy danger
# ---------------------------------------------------------------------------


def _blocked_for_immediate_move(board: Dict, you: Dict, candidate: Point) -> Set[Point]:
    """Cells that block the next immediate move.

    Own tail is free if we do not eat on candidate. Enemy tails stay blocked
    because enemies may eat and keep them.
    """
    foods = _food_set(board)
    will_eat = candidate in foods
    blocked: Set[Point] = set()

    for snake in board.get("snakes", []):
        body = list(_body(snake))
        if not body:
            continue
        if snake.get("id") == you.get("id"):
            if not will_eat:
                body = body[:-1]
        blocked.update(body)
    return blocked



def _blocked_for_planning(board: Dict, you: Dict, assume_our_next: Optional[Point]) -> Set[Point]:
    """Blocked cells for BFS/flood-fill planning.

    If assume_our_next is given, our body is advanced by one step approximately.
    """
    foods = _food_set(board)
    blocked: Set[Point] = set()

    for snake in board.get("snakes", []):
        body = list(_body(snake))
        if not body:
            continue

        if snake.get("id") == you.get("id"):
            if assume_our_next is None:
                # For planning from current head, own tail will usually move.
                body = body[:-1]
            else:
                will_eat = assume_our_next in foods
                body = [assume_our_next] + body
                if not will_eat:
                    body = body[:-1]
        else:
            # Enemy tails are often movable; allow it when enemy probably won't eat.
            if not _has_adjacent_food(_head(snake), foods):
                body = body[:-1]

        blocked.update(body)

    return blocked



def _occupied_cells(snakes: Iterable[Dict]) -> Set[Point]:
    cells: Set[Point] = set()
    for snake in snakes:
        cells.update(_body(snake))
    return cells



def _enemy_head_next_cells(
    board: Dict,
    you: Dict,
    min_len: Optional[int] = None,
    max_len: Optional[int] = None,
) -> Set[Point]:
    width, height = int(board["width"]), int(board["height"])
    cells: Set[Point] = set()
    my_id = you.get("id")

    for snake in board.get("snakes", []):
        if snake.get("id") == my_id:
            continue
        length = int(snake.get("length", len(snake.get("body", []))))
        if min_len is not None and length < min_len:
            continue
        if max_len is not None and length > max_len:
            continue
        h = _head(snake)
        for delta in DIRECTIONS.values():
            p = _add(h, delta)
            if _in_bounds(p, width, height):
                cells.add(p)
    return cells



def _enemy_race_penalty(board: Dict, you: Dict, food: Point, my_dist: int) -> float:
    width, height = int(board["width"]), int(board["height"])
    my_len = int(you["length"])
    my_id = you.get("id")

    enemy_sources: List[Point] = []
    enemy_lengths: Dict[Point, int] = {}
    for snake in board.get("snakes", []):
        if snake.get("id") == my_id:
            continue
        h = _head(snake)
        enemy_sources.append(h)
        enemy_lengths[h] = int(snake.get("length", len(snake.get("body", []))))

    if not enemy_sources:
        return 0.0

    blocked = _occupied_cells(board.get("snakes", []))
    for h in enemy_sources:
        blocked.discard(h)

    # Compute each enemy individually, because length matters.
    penalty = 0.0
    for h in enemy_sources:
        dist_map = _bfs_dist([h], blocked, width, height)
        e_dist = dist_map.get(food)
        if e_dist is None:
            continue
        e_len = enemy_lengths[h]
        if e_len >= my_len and e_dist <= my_dist:
            return BIG_PENALTY
        if e_dist < my_dist:
            penalty += 2_000.0
        elif e_dist == my_dist:
            penalty += 800.0
    return penalty


# ---------------------------------------------------------------------------
# Graph algorithms
# ---------------------------------------------------------------------------


def _bfs_path(start: Point, target: Point, blocked: Set[Point], width: int, height: int) -> Optional[List[Point]]:
    if start == target:
        return [start]
    if target in blocked:
        # Target may be own tail; callers should discard it if desired.
        return None

    q: deque[Point] = deque([start])
    parent: Dict[Point, Optional[Point]] = {start: None}

    while q:
        cur = q.popleft()
        for delta in DIRECTIONS.values():
            nxt = _add(cur, delta)
            if not _in_bounds(nxt, width, height):
                continue
            if nxt in blocked:
                continue
            if nxt in parent:
                continue
            parent[nxt] = cur
            if nxt == target:
                return _reconstruct_path(parent, target)
            q.append(nxt)
    return None



def _reconstruct_path(parent: Dict[Point, Optional[Point]], target: Point) -> List[Point]:
    path: List[Point] = []
    cur: Optional[Point] = target
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path



def _bfs_dist(sources: Iterable[Point], blocked: Set[Point], width: int, height: int) -> Dict[Point, int]:
    q: deque[Point] = deque()
    dist: Dict[Point, int] = {}
    for s in sources:
        if not _in_bounds(s, width, height):
            continue
        q.append(s)
        dist[s] = 0

    while q:
        cur = q.popleft()
        for delta in DIRECTIONS.values():
            nxt = _add(cur, delta)
            if not _in_bounds(nxt, width, height):
                continue
            if nxt in blocked and nxt not in dist:
                continue
            if nxt in dist:
                continue
            dist[nxt] = dist[cur] + 1
            q.append(nxt)
    return dist



def _flood_fill(start: Point, blocked: Set[Point], width: int, height: int, limit: int) -> int:
    if not _in_bounds(start, width, height) or start in blocked:
        return 0
    q: deque[Point] = deque([start])
    seen: Set[Point] = {start}
    while q and len(seen) < limit:
        cur = q.popleft()
        for delta in DIRECTIONS.values():
            nxt = _add(cur, delta)
            if not _in_bounds(nxt, width, height):
                continue
            if nxt in blocked or nxt in seen:
                continue
            seen.add(nxt)
            q.append(nxt)
    return len(seen)



def _reachable(start: Point, target: Point, blocked: Set[Point], width: int, height: int) -> bool:
    return _bfs_path(start, target, blocked, width, height) is not None



def _escape_routes(point: Point, blocked: Set[Point], width: int, height: int) -> int:
    count = 0
    for delta in DIRECTIONS.values():
        p = _add(point, delta)
        if _in_bounds(p, width, height) and p not in blocked:
            count += 1
    return count



def _voronoi_control(board: Dict, you: Dict, candidate: Point, blocked: Set[Point], width: int, height: int) -> int:
    enemies = [s for s in board.get("snakes", []) if s.get("id") != you.get("id")]
    if not enemies:
        return _flood_fill(candidate, blocked - {candidate}, width, height, width * height)

    my_dist = _bfs_dist([candidate], blocked - {candidate}, width, height)
    enemy_heads = [_head(s) for s in enemies]
    enemy_blocked = set(blocked)
    for h in enemy_heads:
        enemy_blocked.discard(h)
    enemy_dist = _bfs_dist(enemy_heads, enemy_blocked, width, height)

    control = 0
    for cell, d in my_dist.items():
        if d < enemy_dist.get(cell, 10_000):
            control += 1
    return control


# ---------------------------------------------------------------------------
# Board parsing helpers
# ---------------------------------------------------------------------------


def _point(p: Dict) -> Point:
    return int(p["x"]), int(p["y"])



def _head(snake: Dict) -> Point:
    return _point(snake["head"])



def _tail(snake: Dict) -> Point:
    return _point(snake["body"][-1])



def _body(snake: Dict) -> List[Point]:
    return [_point(p) for p in snake.get("body", [])]



def _food_set(board: Dict) -> Set[Point]:
    return {_point(f) for f in board.get("food", [])}



def _hazard_set(board: Dict) -> Set[Point]:
    return {_point(h) for h in board.get("hazards", [])}



def _hazard_damage(obj: Dict) -> int:
    # Accept either full game_state or board.
    if "game" in obj:
        ruleset = obj.get("game", {}).get("ruleset", {})
    else:
        ruleset = obj.get("ruleset", {})
    settings = ruleset.get("settings", {}) if isinstance(ruleset, dict) else {}
    try:
        return int(settings.get("hazardDamagePerTurn", 0))
    except Exception:
        return 0



def _has_adjacent_food(head: Point, foods: Set[Point]) -> bool:
    return any(_add(head, d) in foods for d in DIRECTIONS.values())



def _game_id(board: Dict) -> str:
    # choose_move only receives game_state elsewhere, but we only pass board here.
    # Board has no id, so use dimensions + snake ids as a weak key fallback.
    snake_ids = ":".join(sorted(str(s.get("id", "")) for s in board.get("snakes", [])))
    return f"{board.get('width')}x{board.get('height')}:{snake_ids}"



def _clear_target(board: Dict) -> None:
    TARGET_FOOD_BY_GAME.pop(_game_id(board), None)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _add(a: Point, b: Point) -> Point:
    return a[0] + b[0], a[1] + b[1]



def _in_bounds(p: Point, width: int, height: int) -> bool:
    return 0 <= p[0] < width and 0 <= p[1] < height



def _manhattan(a: Point, b: Point) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])



def _dist_to_center(p: Point, width: int, height: int) -> float:
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    return abs(p[0] - cx) + abs(p[1] - cy)
