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
        "head": "beluga",
        "tail": "curled",
        "version": "2.0.0-food-tail-anti-trap",
    }



def choose_move(game_state: Dict) -> str:
    """Main decision function called from /move."""
    try:
        move = choose_move_safe(game_state)
        if move in DIRECTIONS:
            return move
    except Exception:
        # Never let /move fail in tournament mode.
        pass
    return fallback_move(game_state)


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
    elif mode == "control":
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
    if health <= HUNGRY_HEALTH:
        return "food"
    if length < GROW_UNTIL_LENGTH:
        return "grow"

    blocked = _blocked_for_planning(board, you, assume_our_next=None)
    blocked.discard(head)
    current_room = _flood_fill(head, blocked, width, height, limit=width * height)

    # If already cramped, do not think about fancy attack/food. Survive.
    if current_room < max(length * 2, length + 8):
        return "survival"

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

        # Attack only if it does not break safety.
        if p in smaller_head_next and room >= length * 2 and reaches_tail:
            score += 3_000.0
            reasons["attack"] = 1.0
        else:
            reasons["attack"] = 0.0

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
    try:
        board = game_state["board"]
        you = game_state["you"]
        width, height = int(board["width"]), int(board["height"])
        head = _head(you)
        occupied = _occupied_cells(board.get("snakes", []))
        for move, delta in DIRECTIONS.items():
            p = _add(head, delta)
            if _in_bounds(p, width, height) and p not in occupied:
                return move
    except Exception:
        pass
    return "up"


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
