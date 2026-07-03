"""Strong heuristic move-selection logic for Battlesnake.

Drop-in replacement for the baseline `logic.py`.

Main ideas:
- never crash: `choose_move` always returns one of up/down/left/right;
- treat our own tail correctly when it will move away;
- avoid instant death, small pockets, bad head-to-heads and hazards;
- eat only when food is safe / needed;
- prefer territory, tail reachability, escape routes, and center control;
- add a cheap 1-ply safety check: after this move, will we still have moves?

Coordinate system: (0, 0) is bottom-left.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

Point = Tuple[int, int]

DIRECTIONS: Dict[str, Point] = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}

# Tunable constants. Start conservative; tune with local games.
MAX_HEALTH = 100
VERY_HUNGRY_HEALTH = 25
HUNGRY_HEALTH = 45
OK_HEALTH = 70

DEATH_SCORE = -1_000_000.0
H2H_FATAL_PENALTY = 120_000.0
SMALL_POCKET_PENALTY = 40_000.0
NO_NEXT_MOVE_PENALTY = 35_000.0
ONE_EXIT_PENALTY = 2_500.0
HAZARD_BASE_PENALTY = 900.0


@dataclass(frozen=True)
class MoveScore:
    move: str
    score: float
    reasons: Dict[str, float]


def get_info() -> Dict[str, str]:
    """Appearance + metadata returned from GET /."""
    return {
        "apiversion": "1",
        "author": "hackathon",
        "color": "#6434eb",
        "head": "smart-caterpillar",
        "tail": "weight",
        "version": "1.0.0-heuristic",
    }


def choose_move(game_state: Dict) -> str:
    """Return the next move for the current turn.

    This function is intentionally defensive: any unexpected issue falls back to
    a basic legal-ish move so the server never returns 500 on /move.
    """
    try:
        scores = score_moves(game_state)
        if scores:
            return max(scores, key=lambda s: s.score).move
    except Exception:
        # In a tournament, returning any move is better than timing out / 500.
        pass

    return fallback_move(game_state)


def score_moves(game_state: Dict) -> List[MoveScore]:
    board = game_state["board"]
    you = game_state["you"]

    width = int(board["width"])
    height = int(board["height"])
    foods = _food_set(board)
    hazards = _hazard_set(board)
    hazard_damage = _hazard_damage(game_state)

    my_id = you["id"]
    my_head = _head(you)
    my_len = int(you["length"])
    my_health = int(you["health"])

    enemies = [s for s in board.get("snakes", []) if s.get("id") != my_id]
    bigger_or_equal_threat = _enemy_head_next_cells(board, you, min_len=my_len)
    smaller_enemy_head_cells = _enemy_head_next_cells(board, you, max_len=my_len - 1)

    mode = _mode(board, you)
    results: List[MoveScore] = []

    for move, delta in DIRECTIONS.items():
        nxt = _add(my_head, delta)
        reasons: Dict[str, float] = {}

        # 1) Immediate hard safety.
        if not _in_bounds(nxt, width, height):
            results.append(MoveScore(move, DEATH_SCORE, {"death_wall": 1.0}))
            continue

        occupied_now = _occupied_for_direct_move(board, you, nxt)
        if nxt in occupied_now:
            results.append(MoveScore(move, DEATH_SCORE, {"death_body": 1.0}))
            continue

        # If hazard would kill us before possible food recovery, avoid it.
        hazard_cost = hazard_damage if nxt in hazards else 0
        will_eat = nxt in foods
        if hazard_cost and not will_eat and my_health - 1 - hazard_cost <= 0:
            results.append(MoveScore(move, DEATH_SCORE, {"death_hazard": 1.0}))
            continue

        score = 0.0

        # 2) Occupancy used for planning after our candidate move.
        planning_blocked = _occupied_for_planning(board, you, nxt)
        planning_blocked.discard(nxt)

        # 3) Space / traps.
        open_space = _flood_fill(nxt, planning_blocked, width, height, limit=width * height)
        reasons["open_space"] = float(open_space)
        score += 22.0 * open_space

        space_ratio = open_space / max(1, my_len)
        reasons["space_ratio"] = space_ratio
        if open_space < my_len:
            score -= SMALL_POCKET_PENALTY * (1.0 - open_space / max(1, my_len))
        elif open_space < my_len * 1.5:
            score -= 2_000.0

        # 4) Escape routes and 1-ply next safety.
        escape_routes = _escape_routes(nxt, planning_blocked, width, height)
        reasons["escape_routes"] = float(escape_routes)
        score += 450.0 * escape_routes
        if escape_routes == 0:
            score -= NO_NEXT_MOVE_PENALTY
        elif escape_routes == 1:
            score -= ONE_EXIT_PENALTY

        next_safe_count = _next_safe_move_count(board, you, nxt)
        reasons["next_safe_count"] = float(next_safe_count)
        score += 550.0 * next_safe_count
        if next_safe_count == 0:
            score -= NO_NEXT_MOVE_PENALTY

        # 5) Tail reachability: strong anti-trap signal.
        tail = _tail(you)
        reaches_tail = _reachable(nxt, tail, planning_blocked - {tail}, width, height)
        reasons["reaches_tail"] = 1.0 if reaches_tail else 0.0
        score += 3_500.0 if reaches_tail else -2_500.0

        # 6) Head-to-head risk / attack opportunity.
        if nxt in bigger_or_equal_threat:
            reasons["h2h_fatal_risk"] = 1.0
            score -= H2H_FATAL_PENALTY
        else:
            reasons["h2h_fatal_risk"] = 0.0

        if nxt in smaller_enemy_head_cells and open_space >= my_len:
            reasons["attack_smaller"] = 1.0
            score += 2_500.0
        else:
            reasons["attack_smaller"] = 0.0

        # Nearby bigger heads are dangerous even if not direct h2h.
        bigger_heads = [_head(s) for s in enemies if int(s["length"]) >= my_len]
        nearest_bigger_head = min((_manhattan(nxt, h) for h in bigger_heads), default=width + height)
        reasons["nearest_bigger_head"] = float(nearest_bigger_head)
        if nearest_bigger_head == 1:
            score -= 4_500.0
        elif nearest_bigger_head == 2:
            score -= 800.0

        # 7) Food: safe and context-dependent.
        food_score = _food_score(
            board=board,
            you=you,
            candidate=nxt,
            blocked=planning_blocked,
            width=width,
            height=height,
            mode=mode,
        )
        reasons["food_score"] = food_score
        score += food_score

        if will_eat:
            # Eating is great when hungry, but neutral/bad when healthy and cramped.
            if my_health <= VERY_HUNGRY_HEALTH:
                score += 8_000.0
            elif my_health <= HUNGRY_HEALTH:
                score += 3_000.0
            elif open_space < my_len * 2:
                score -= 1_500.0
            else:
                score += 250.0

        # 8) Territory / Voronoi control.
        voronoi = _voronoi_control(board, you, nxt, planning_blocked, width, height)
        reasons["voronoi"] = float(voronoi)
        score += 4.0 * voronoi

        # 9) Hazards.
        if nxt in hazards:
            reasons["hazard"] = 1.0
            hazard_penalty = HAZARD_BASE_PENALTY + 45.0 * hazard_damage
            if my_health <= OK_HEALTH:
                hazard_penalty *= 2.0
            if my_health <= HUNGRY_HEALTH:
                hazard_penalty *= 2.0
            score -= hazard_penalty
        else:
            reasons["hazard"] = 0.0

        # 10) Center / wall preference. Small bonus, never dominates safety.
        center_dist = _dist_to_center(nxt, width, height)
        reasons["center_dist"] = center_dist
        score -= 12.0 * center_dist

        wall_dist = min(nxt[0], width - 1 - nxt[0], nxt[1], height - 1 - nxt[1])
        reasons["wall_dist"] = float(wall_dist)
        if wall_dist == 0 and mode != "survival":
            score -= 250.0
        else:
            score += 30.0 * wall_dist

        # 11) Mode-specific shaping.
        if mode == "hungry":
            score += 0.5 * food_score
        elif mode == "attack":
            score += 700.0 if reasons["attack_smaller"] else 0.0
            score += 1.5 * voronoi
        elif mode == "survival":
            score += 35.0 * open_space
            score += 2_000.0 if reaches_tail else -1_000.0
        elif mode == "control":
            score += 2.0 * voronoi
            score += 8.0 * open_space

        reasons["mode"] = {"hungry": 1, "survival": 2, "attack": 3, "control": 4}.get(mode, 0)
        results.append(MoveScore(move, score, reasons))

    return results


def fallback_move(game_state: Dict) -> str:
    """Simple safe fallback used only if scoring fails."""
    try:
        board = game_state["board"]
        you = game_state["you"]
        width, height = int(board["width"]), int(board["height"])
        head = _head(you)
        occupied = _occupied_cells(board.get("snakes", []))

        for move, delta in DIRECTIONS.items():
            nxt = _add(head, delta)
            if _in_bounds(nxt, width, height) and nxt not in occupied:
                return move
    except Exception:
        pass
    return "up"


# ---------------------------------------------------------------------------
# Strategy helpers
# ---------------------------------------------------------------------------


def _mode(board: Dict, you: Dict) -> str:
    health = int(you["health"])
    my_len = int(you["length"])
    enemies = [s for s in board.get("snakes", []) if s.get("id") != you.get("id")]

    if health <= HUNGRY_HEALTH:
        return "hungry"

    # If we are clearly bigger than at least one nearby enemy, attack can pay off.
    my_head = _head(you)
    smaller_nearby = any(
        int(e["length"]) < my_len and _manhattan(my_head, _head(e)) <= 4
        for e in enemies
    )
    if smaller_nearby and my_len >= 4:
        return "attack"

    # If current available space is tight, prioritize survival.
    occupied = _occupied_for_planning(board, you, my_head)
    occupied.discard(my_head)
    space = _flood_fill(my_head, occupied, int(board["width"]), int(board["height"]), limit=my_len * 2)
    if space < my_len * 1.5:
        return "survival"

    return "control"


def _food_score(
    board: Dict,
    you: Dict,
    candidate: Point,
    blocked: Set[Point],
    width: int,
    height: int,
    mode: str,
) -> float:
    foods = list(_food_set(board))
    if not foods:
        return 0.0

    health = int(you["health"])
    my_len = int(you["length"])
    enemies = [s for s in board.get("snakes", []) if s.get("id") != you.get("id")]

    my_dist = _bfs_dist([candidate], blocked, width, height)
    enemy_heads = [_head(s) for s in enemies]
    enemy_blocked = _occupied_for_enemy_distance(board, you)
    enemy_dist = _bfs_dist(enemy_heads, enemy_blocked, width, height) if enemy_heads else {}

    best = 0.0
    for food in foods:
        d = my_dist.get(food)
        if d is None:
            continue

        # Need to be able to survive until the food. Eating at health=1 is OK if d==0.
        if d >= health:
            continue

        # Avoid food that a bigger/equal enemy can contest first or at the same time.
        contested = False
        for enemy in enemies:
            e_len = int(enemy["length"])
            e_head = _head(enemy)
            e_d = enemy_dist.get(food, _manhattan(e_head, food))
            if e_len >= my_len and e_d <= d:
                contested = True
                break
        if contested:
            continue

        # Check that the food cell is not a tiny dead-end.
        space_after_food = _flood_fill(food, blocked, width, height, limit=max(my_len + 3, 12))
        if space_after_food < min(my_len, 8):
            continue

        closeness = max(0, width + height - d)
        if health <= VERY_HUNGRY_HEALTH:
            value = 900.0 * closeness - 40.0 * d
        elif health <= HUNGRY_HEALTH:
            value = 350.0 * closeness - 25.0 * d
        elif mode == "hungry":
            value = 220.0 * closeness - 20.0 * d
        elif health <= OK_HEALTH and d <= 5:
            value = 140.0 * closeness - 10.0 * d
        else:
            # Healthy snakes should not over-prioritize food and become huge/trapped.
            value = 35.0 * closeness - 5.0 * d

        best = max(best, value)

    return best


def _voronoi_control(board: Dict, you: Dict, candidate: Point, blocked: Set[Point], width: int, height: int) -> int:
    enemies = [s for s in board.get("snakes", []) if s.get("id") != you.get("id")]
    if not enemies:
        return width * height

    my_dist = _bfs_dist([candidate], blocked, width, height)
    enemy_heads = [_head(s) for s in enemies]
    enemy_dist = _bfs_dist(enemy_heads, blocked, width, height)

    control = 0
    for cell, md in my_dist.items():
        if md < enemy_dist.get(cell, 10_000):
            control += 1
    return control


def _next_safe_move_count(board: Dict, you: Dict, candidate: Point) -> int:
    """Cheap approximation: after we move to candidate, how many exits exist?"""
    width, height = int(board["width"]), int(board["height"])
    foods = _food_set(board)
    will_eat = candidate in foods

    # Build an approximate body after our move.
    my_body = [_point(p) for p in you["body"]]
    new_my_body = [candidate] + my_body
    if not will_eat:
        new_my_body = new_my_body[:-1]

    blocked: Set[Point] = set(new_my_body)
    for snake in board.get("snakes", []):
        if snake.get("id") == you.get("id"):
            continue
        # Conservative for enemies: keep their full body occupied.
        blocked.update(_body(snake))

    blocked.discard(candidate)

    danger = _enemy_head_next_cells(board, you, min_len=int(you["length"]))
    count = 0
    for delta in DIRECTIONS.values():
        nxt = _add(candidate, delta)
        if not _in_bounds(nxt, width, height):
            continue
        if nxt in blocked:
            continue
        if nxt in danger:
            continue
        count += 1
    return count


# ---------------------------------------------------------------------------
# Occupancy and danger maps
# ---------------------------------------------------------------------------


def _occupied_cells(snakes: Iterable[Dict]) -> Set[Point]:
    occupied: Set[Point] = set()
    for snake in snakes:
        occupied.update(_body(snake))
    return occupied


def _occupied_for_direct_move(board: Dict, you: Dict, candidate: Point) -> Set[Point]:
    """Cells that should block the immediate candidate move.

    Our own tail is allowed if it will move away this turn. Enemy tails remain
    blocked for direct moves because an enemy may eat and keep its tail.
    """
    foods = _food_set(board)
    will_eat = candidate in foods

    occupied: Set[Point] = set()
    for snake in board.get("snakes", []):
        body = _body(snake)
        if snake.get("id") == you.get("id"):
            if not will_eat and body:
                body = body[:-1]
        occupied.update(body)
    return occupied


def _occupied_for_planning(board: Dict, you: Dict, candidate: Point) -> Set[Point]:
    """Cells treated as blocked for flood-fill/planning.

    Slightly less conservative than direct collision: own tail may move away;
    enemy tails are also often movable, but we keep most enemy bodies solid.
    """
    foods = _food_set(board)
    will_eat = candidate in foods

    occupied: Set[Point] = set()
    for snake in board.get("snakes", []):
        body = _body(snake)
        if not body:
            continue
        if snake.get("id") == you.get("id"):
            if not will_eat:
                body = body[:-1]
        else:
            # Planning can ignore enemy tail only when no food is adjacent to its head;
            # otherwise it may eat and keep the tail.
            if not _has_adjacent_food(_head(snake), foods):
                body = body[:-1]
        occupied.update(body)
    return occupied


def _occupied_for_enemy_distance(board: Dict, you: Dict) -> Set[Point]:
    """Blocked cells for estimating enemy distance to food."""
    occupied = _occupied_cells(board.get("snakes", []))
    # Do not block enemy heads as BFS sources; remove all heads from blocked.
    for snake in board.get("snakes", []):
        occupied.discard(_head(snake))
    return occupied


def _enemy_head_next_cells(
    board: Dict,
    you: Dict,
    min_len: Optional[int] = None,
    max_len: Optional[int] = None,
) -> Set[Point]:
    """Cells enemy heads may enter next turn, filtered by enemy length."""
    width, height = int(board["width"]), int(board["height"])
    cells: Set[Point] = set()

    for snake in board.get("snakes", []):
        if snake.get("id") == you.get("id"):
            continue
        length = int(snake["length"])
        if min_len is not None and length < min_len:
            continue
        if max_len is not None and length > max_len:
            continue

        head = _head(snake)
        for delta in DIRECTIONS.values():
            nxt = _add(head, delta)
            if _in_bounds(nxt, width, height):
                cells.add(nxt)
    return cells


# ---------------------------------------------------------------------------
# Graph algorithms
# ---------------------------------------------------------------------------


def _flood_fill(start: Point, blocked: Set[Point], width: int, height: int, limit: int) -> int:
    if start in blocked or not _in_bounds(start, width, height):
        return 0

    seen: Set[Point] = {start}
    dq: deque[Point] = deque([start])
    count = 0

    while dq and count < limit:
        cur = dq.popleft()
        count += 1
        for delta in DIRECTIONS.values():
            nb = _add(cur, delta)
            if nb in seen or nb in blocked or not _in_bounds(nb, width, height):
                continue
            seen.add(nb)
            dq.append(nb)

    return count


def _bfs_dist(sources: Iterable[Point], blocked: Set[Point], width: int, height: int) -> Dict[Point, int]:
    dist: Dict[Point, int] = {}
    dq: deque[Point] = deque()

    for source in sources:
        if not _in_bounds(source, width, height):
            continue
        if source in dist:
            continue
        dist[source] = 0
        dq.append(source)

    while dq:
        cur = dq.popleft()
        for delta in DIRECTIONS.values():
            nb = _add(cur, delta)
            if nb in dist or nb in blocked or not _in_bounds(nb, width, height):
                continue
            dist[nb] = dist[cur] + 1
            dq.append(nb)

    return dist


def _reachable(start: Point, target: Point, blocked: Set[Point], width: int, height: int) -> bool:
    if start == target:
        return True
    if start in blocked:
        return False

    seen: Set[Point] = {start}
    dq: deque[Point] = deque([start])

    while dq:
        cur = dq.popleft()
        for delta in DIRECTIONS.values():
            nb = _add(cur, delta)
            if nb == target:
                return True
            if nb in seen or nb in blocked or not _in_bounds(nb, width, height):
                continue
            seen.add(nb)
            dq.append(nb)
    return False


def _escape_routes(point: Point, blocked: Set[Point], width: int, height: int) -> int:
    count = 0
    for delta in DIRECTIONS.values():
        nb = _add(point, delta)
        if _in_bounds(nb, width, height) and nb not in blocked:
            count += 1
    return count


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _point(obj: Dict) -> Point:
    return int(obj["x"]), int(obj["y"])


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


def _hazard_damage(game_state: Dict) -> int:
    try:
        return int(game_state.get("game", {}).get("ruleset", {}).get("settings", {}).get("hazardDamagePerTurn", 0))
    except Exception:
        return 0


def _has_adjacent_food(head: Point, foods: Set[Point]) -> bool:
    return any(_add(head, delta) in foods for delta in DIRECTIONS.values())


def _add(a: Point, b: Point) -> Point:
    return a[0] + b[0], a[1] + b[1]


def _in_bounds(p: Point, width: int, height: int) -> bool:
    return 0 <= p[0] < width and 0 <= p[1] < height


def _manhattan(a: Point, b: Point) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _dist_to_center(p: Point, width: int, height: int) -> float:
    return abs(p[0] - (width - 1) / 2.0) + abs(p[1] - (height - 1) / 2.0)