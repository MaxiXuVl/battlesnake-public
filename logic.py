"""Improved heuristic move-selection logic for Battlesnake.

Drop-in replacement for logic.py.

Design goals:
- Never fail to return a legal move.
- Prefer survival over greed.
- Use flood fill, tail reachability, safe food, head-to-head risk,
  attacking smaller snakes, Voronoi-like territory, and a lightweight
  one-step lookahead.

Board coordinates: (0, 0) is the bottom-left corner.
"""

from __future__ import annotations

from collections import deque
from itertools import product
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

Point = Tuple[int, int]

DIRECTIONS: Dict[str, Point] = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}

# Tunable constants. Start with these, then tune from local games/logs.
VERY_BAD = -1_000_000.0
DEATH_PENALTY = 100_000.0
HEAD_TO_HEAD_PENALTY = 25_000.0
TRAP_PENALTY = 8_000.0
LOW_ESCAPE_PENALTY = 700.0
TAIL_REACH_BONUS = 1_800.0
FOOD_EAT_BONUS = 900.0
SAFE_FOOD_BONUS = 450.0
ATTACK_BONUS = 850.0
CENTER_BONUS = 12.0
SPACE_WEIGHT = 28.0
VORONOI_WEIGHT = 2.2
LOOKAHEAD_WEIGHT = 0.35

# Food strategy.
PANIC_HEALTH = 25
HUNGRY_HEALTH = 45
COMFORTABLE_HEALTH = 70

# Enemy move combinations can explode with many snakes. This cap keeps response fast.
MAX_ENEMY_COMBINATIONS = 64


def get_info() -> Dict[str, str]:
    """Appearance + metadata returned from GET /."""
    return {
        "apiversion": "1",
        "author": "hackathon",
        "color": "#6434eb",
        "head": "smart-caterpillar",
        "tail": "weight",
        "version": "0.2.0-heuristic",
    }


def choose_move(game_state: Dict) -> str:
    """Return the next move.

    This function is intentionally defensive: any unexpected issue falls back to
    a simple legal move so the server never crashes during a game.
    """
    try:
        move = choose_move_heuristic(game_state)
        if move:
            return move
    except Exception:  # noqa: BLE001 - Battlesnake must always answer
        pass

    return _emergency_move(game_state)


def choose_move_heuristic(game_state: Dict) -> Optional[str]:
    """Score legal moves and return the best one."""
    board = game_state["board"]
    you = game_state["you"]
    width, height = board["width"], board["height"]
    my_head = _point(you["head"])
    my_length = int(you["length"])
    health = int(you["health"])
    foods = {_point(f) for f in board.get("food", [])}

    candidates: List[Tuple[str, float, Dict[str, float]]] = []

    for move, delta in DIRECTIONS.items():
        nxt = _add(my_head, delta)
        features = _evaluate_move(game_state, move)
        score = _score_features(features, health, my_length, width, height)

        # A small paranoid lookahead: if enemies have obvious replies, evaluate
        # the worst few outcomes. This catches many head-to-head and trap cases.
        if score > VERY_BAD / 2:
            score += LOOKAHEAD_WEIGHT * _one_step_lookahead(game_state, move)

        candidates.append((move, score, features))

    if not candidates:
        return None

    # Prefer deterministic behavior: max by score, then a stable direction order.
    direction_order = {m: i for i, m in enumerate(["up", "right", "down", "left"])}
    candidates.sort(key=lambda x: (x[1], -direction_order.get(x[0], 99)), reverse=True)

    # If every move is terrible, still return the least terrible legal-ish move.
    best_move, _best_score, _features = candidates[0]
    return best_move


def _evaluate_move(game_state: Dict, move: str) -> Dict[str, float]:
    """Compute robust handcrafted features for one candidate move."""
    board = game_state["board"]
    you = game_state["you"]
    width, height = board["width"], board["height"]
    snakes = board.get("snakes", [])
    foods = {_point(f) for f in board.get("food", [])}

    my_id = you["id"]
    my_head = _point(you["head"])
    my_tail = _point(you["body"][-1])
    my_length = int(you["length"])
    health = int(you["health"])
    nxt = _add(my_head, DIRECTIONS[move])

    features: Dict[str, float] = {
        "in_bounds": 1.0 if _in_bounds(nxt, width, height) else 0.0,
        "body_collision": 0.0,
        "head_to_head_danger": 0.0,
        "open_space": 0.0,
        "space_ratio": 0.0,
        "trap_risk": 0.0,
        "escape_routes": 0.0,
        "reaches_tail": 0.0,
        "safe_food_bonus": 0.0,
        "eat_food": 1.0 if nxt in foods else 0.0,
        "food_dist": float(width + height + 100),
        "food_is_safe": 0.0,
        "attack_bonus": 0.0,
        "voronoi": 0.0,
        "center_score": 0.0,
        "wall_penalty": 0.0,
        "future_safe_moves": 0.0,
    }

    if not _in_bounds(nxt, width, height):
        return features

    # Occupancy for this exact move. We allow our own tail if we do not eat,
    # because it moves away at the same turn. Enemy tails are treated mostly as
    # occupied unless we know they are not growing.
    occupied = _occupied_cells_for_my_candidate(board, you, nxt)
    if nxt in occupied:
        features["body_collision"] = 1.0
        return features

    enemy_head_danger = _enemy_head_danger_cells(snakes, my_id, my_length, width, height)
    if nxt in enemy_head_danger:
        features["head_to_head_danger"] = 1.0

    # Space after the move. Use occupied plus our new head.
    future_blocked = set(occupied)
    future_blocked.add(nxt)
    # But flood fill starts on nxt; it should not block itself.
    future_blocked.discard(nxt)

    max_cells = width * height
    open_space = _flood_fill(nxt, future_blocked, width, height, limit=max_cells)
    features["open_space"] = float(open_space)
    features["space_ratio"] = float(open_space / max(my_length, 1))
    if open_space < my_length:
        features["trap_risk"] = 1.0

    escape_routes = _count_escape_routes(nxt, future_blocked, width, height)
    features["escape_routes"] = float(escape_routes)

    # Tail reachability: important anti-trap heuristic.
    tail_blocked = set(future_blocked)
    tail_blocked.discard(my_tail)
    features["reaches_tail"] = 1.0 if _reachable(nxt, my_tail, tail_blocked, width, height) else 0.0

    # Safe food evaluation.
    food_eval = _evaluate_food(game_state, nxt, future_blocked)
    features.update(food_eval)

    # Attack smaller snakes if we are longer, but only if the move is otherwise safe.
    features["attack_bonus"] = float(_attack_opportunity(board, you, nxt, future_blocked))

    # Voronoi-like territory: cells we can reach before enemies.
    features["voronoi"] = float(_voronoi_control(board, you, nxt, future_blocked))

    # Center is mildly good early/mid-game; walls are only bad if they reduce exits.
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    dist_to_center = abs(nxt[0] - cx) + abs(nxt[1] - cy)
    features["center_score"] = float(width + height - dist_to_center)

    wall_dist = min(nxt[0], width - 1 - nxt[0], nxt[1], height - 1 - nxt[1])
    features["wall_penalty"] = 1.0 if wall_dist == 0 and escape_routes <= 2 else 0.0

    # Quick future safety: after we move, how many non-suicidal moves remain?
    features["future_safe_moves"] = float(_future_safe_move_count(board, you, nxt, future_blocked))

    return features


def _score_features(features: Dict[str, float], health: int, my_length: int, width: int, height: int) -> float:
    """Convert features to a single score."""
    if not features["in_bounds"]:
        return VERY_BAD
    if features["body_collision"]:
        return VERY_BAD + 1_000

    score = 0.0

    # Survival first.
    score += SPACE_WEIGHT * features["open_space"]
    score += VORONOI_WEIGHT * features["voronoi"]
    score += 220.0 * features["escape_routes"]
    score += 400.0 * features["future_safe_moves"]

    if features["reaches_tail"]:
        score += TAIL_REACH_BONUS
    else:
        score -= 1_000.0

    if features["trap_risk"]:
        # The smaller the space ratio, the more dangerous.
        score -= TRAP_PENALTY * max(1.0, 1.5 - features["space_ratio"])

    if features["escape_routes"] <= 0:
        score -= DEATH_PENALTY
    elif features["escape_routes"] == 1:
        score -= LOW_ESCAPE_PENALTY

    if features["head_to_head_danger"]:
        score -= HEAD_TO_HEAD_PENALTY

    # Food: aggressive only when needed; otherwise eating is optional and may hurt mobility.
    if health <= PANIC_HEALTH:
        score += 1_400.0 * features["safe_food_bonus"]
        score += FOOD_EAT_BONUS * features["eat_food"]
    elif health <= HUNGRY_HEALTH:
        score += 900.0 * features["safe_food_bonus"]
        score += 0.6 * FOOD_EAT_BONUS * features["eat_food"]
    elif health <= COMFORTABLE_HEALTH:
        score += 350.0 * features["safe_food_bonus"]
        score += 120.0 * features["eat_food"]
    else:
        # Do not force food when healthy; small bonus if it is already safe and free.
        score += 80.0 * features["safe_food_bonus"]
        score -= 80.0 * features["eat_food"]

    # Attack only after survival signals are acceptable.
    if features["open_space"] >= max(my_length, 6) and features["escape_routes"] >= 2:
        score += ATTACK_BONUS * features["attack_bonus"]

    # Mild positional preferences.
    score += CENTER_BONUS * features["center_score"]
    score -= 600.0 * features["wall_penalty"]

    return score


def _one_step_lookahead(game_state: Dict, my_move: str) -> float:
    """Approximate worst-case safety after our move and possible enemy moves.

    This is intentionally lightweight. It does not fully simulate eating/growth for
    every snake; it mostly asks: can enemies occupy/contest our next head, and will
    our resulting position still have space?
    """
    board = game_state["board"]
    you = game_state["you"]
    width, height = board["width"], board["height"]
    my_id = you["id"]
    my_length = int(you["length"])
    my_next = _add(_point(you["head"]), DIRECTIONS[my_move])

    if not _in_bounds(my_next, width, height):
        return -DEATH_PENALTY

    enemy_moves: List[List[Point]] = []
    for snake in board.get("snakes", []):
        if snake["id"] == my_id:
            continue
        moves = _possible_enemy_next_heads(board, snake)
        if moves:
            enemy_moves.append(moves[:4])

    if not enemy_moves:
        return 0.0

    worst = 10_000.0
    checked = 0

    for combo in product(*enemy_moves):
        checked += 1
        if checked > MAX_ENEMY_COMBINATIONS:
            break

        enemy_next_heads = set(combo)
        penalty = 0.0

        # If an equal/bigger enemy can move into our new head, we may die.
        for snake, enemy_next in zip([s for s in board["snakes"] if s["id"] != my_id], combo):
            if enemy_next == my_next and int(snake["length"]) >= my_length:
                penalty -= HEAD_TO_HEAD_PENALTY

        # If enemies occupy our exits, reduce future mobility.
        blocked = _occupied_cells_for_my_candidate(board, you, my_next)
        blocked.update(enemy_next_heads)
        blocked.discard(my_next)
        exits = _count_escape_routes(my_next, blocked, width, height)
        space = _flood_fill(my_next, blocked, width, height, limit=width * height)

        outcome = 250.0 * exits + 12.0 * space + penalty
        worst = min(worst, outcome)

    return worst


def _evaluate_food(game_state: Dict, start: Point, blocked: Set[Point]) -> Dict[str, float]:
    """Return food-related features from candidate next cell."""
    board = game_state["board"]
    you = game_state["you"]
    width, height = board["width"], board["height"]
    foods = [_point(f) for f in board.get("food", [])]
    health = int(you["health"])
    my_length = int(you["length"])

    result = {
        "food_dist": float(width + height + 100),
        "food_is_safe": 0.0,
        "safe_food_bonus": 0.0,
    }
    if not foods:
        return result

    dist_map = _bfs_dist([start], blocked, width, height)

    best_bonus = 0.0
    best_dist = width + height + 100
    best_safe = 0.0

    enemy_dists = _enemy_distance_maps(board, you["id"], blocked)

    for food in foods:
        if food not in dist_map:
            continue
        my_dist = dist_map[food]
        best_dist = min(best_dist, my_dist)

        # Can we reach it before starving?
        if my_dist >= health:
            continue

        # Is an enemy likely to arrive first or at the same time?
        enemy_best = min((dm.get(food, 10_000) for dm in enemy_dists), default=10_000)
        enemy_can_contest = enemy_best <= my_dist

        # After reaching food, there should be some room around it.
        space_near_food = _flood_fill(food, blocked - {food}, width, height, limit=max(my_length + 3, 12))
        enough_space = space_near_food >= min(my_length, 8)

        if enemy_can_contest and health > PANIC_HEALTH:
            continue
        if not enough_space and health > PANIC_HEALTH:
            continue

        # Closer food is better; panic makes food much more valuable.
        urgency = 2.2 if health <= PANIC_HEALTH else 1.2 if health <= HUNGRY_HEALTH else 0.35
        bonus = urgency * max(0.0, float(width + height - my_dist))
        if enough_space:
            bonus += 4.0
        if not enemy_can_contest:
            bonus += 6.0

        if bonus > best_bonus:
            best_bonus = bonus
            best_safe = 1.0

    result["food_dist"] = float(best_dist)
    result["food_is_safe"] = best_safe
    result["safe_food_bonus"] = float(best_bonus)
    return result


def _attack_opportunity(board: Dict, you: Dict, nxt: Point, blocked: Set[Point]) -> float:
    """Positive signal when moving near smaller enemy heads safely."""
    my_id = you["id"]
    my_length = int(you["length"])
    width, height = board["width"], board["height"]
    bonus = 0.0

    for snake in board.get("snakes", []):
        if snake["id"] == my_id:
            continue
        enemy_len = int(snake["length"])
        if enemy_len >= my_length:
            continue
        ehead = _point(snake["head"])
        dist = _manhattan(nxt, ehead)

        # Adjacent to smaller head: pressure. Same cell is handled by h2h rules.
        if dist == 1:
            bonus += 1.0
        elif dist == 2:
            bonus += 0.35

        # Cutting off smaller snake's exits is valuable.
        exits = _count_escape_routes(ehead, blocked, width, height)
        if exits <= 1 and dist <= 2:
            bonus += 0.6

    return bonus


def _voronoi_control(board: Dict, you: Dict, my_start: Point, blocked: Set[Point]) -> int:
    """Count cells we reach strictly before any enemy head."""
    width, height = board["width"], board["height"]
    my_id = you["id"]
    my_dist = _bfs_dist([my_start], blocked, width, height)

    enemy_heads = [
        _point(s["head"])
        for s in board.get("snakes", [])
        if s["id"] != my_id and _in_bounds(_point(s["head"]), width, height)
    ]
    if not enemy_heads:
        return len(my_dist)

    enemy_dist = _bfs_dist(enemy_heads, blocked, width, height)
    return sum(1 for cell, d in my_dist.items() if d < enemy_dist.get(cell, 10_000))


def _future_safe_move_count(board: Dict, you: Dict, my_next: Point, blocked: Set[Point]) -> int:
    """Approximate how many safe moves we will have after this move."""
    width, height = board["width"], board["height"]
    my_id = you["id"]
    my_length = int(you["length"])
    danger = _enemy_head_danger_cells(board.get("snakes", []), my_id, my_length, width, height)

    count = 0
    for delta in DIRECTIONS.values():
        nb = _add(my_next, delta)
        if not _in_bounds(nb, width, height):
            continue
        if nb in blocked:
            continue
        if nb in danger:
            continue
        count += 1
    return count


def _possible_enemy_next_heads(board: Dict, enemy: Dict) -> List[Point]:
    """Possible next head cells for an enemy using conservative occupancy."""
    width, height = board["width"], board["height"]
    head = _point(enemy["head"])
    occupied = _occupied_cells(board.get("snakes", []), include_tails=False)
    result = []
    for delta in DIRECTIONS.values():
        nxt = _add(head, delta)
        if _in_bounds(nxt, width, height) and nxt not in occupied:
            result.append(nxt)
    return result


def _enemy_head_danger_cells(
    snakes: Sequence[Dict],
    my_id: str,
    my_length: int,
    width: int,
    height: int,
) -> Set[Point]:
    """Cells where equal-or-larger enemies could move their heads next turn."""
    danger: Set[Point] = set()
    for snake in snakes:
        if snake["id"] == my_id:
            continue
        if int(snake["length"]) < my_length:
            continue
        ehead = _point(snake["head"])
        for delta in DIRECTIONS.values():
            nxt = _add(ehead, delta)
            if _in_bounds(nxt, width, height):
                danger.add(nxt)
    return danger


def _occupied_cells(snakes: Sequence[Dict], include_tails: bool = True) -> Set[Point]:
    """Cells occupied by snake bodies."""
    occupied: Set[Point] = set()
    for snake in snakes:
        body = [_point(seg) for seg in snake.get("body", [])]
        if not include_tails and body:
            body = body[:-1]
        occupied.update(body)
    return occupied


def _occupied_cells_for_my_candidate(board: Dict, you: Dict, my_next: Point) -> Set[Point]:
    """Occupied cells for evaluating our candidate next head.

    Our own tail is free if we are not eating on this move. Enemy tails are also
    allowed to clear when their current tail is not food; this is a pragmatic
    compromise between safety and mobility.
    """
    foods = {_point(f) for f in board.get("food", [])}
    my_id = you["id"]
    occupied: Set[Point] = set()

    for snake in board.get("snakes", []):
        body = [_point(seg) for seg in snake.get("body", [])]
        if not body:
            continue

        if snake["id"] == my_id:
            # If we eat, our tail does not move this turn.
            if my_next not in foods:
                body = body[:-1]
        else:
            # Enemy may eat, but if their tail is not currently on food, treating
            # it as free improves mobility. Head danger still protects us.
            tail = body[-1]
            if tail not in foods:
                body = body[:-1]

        occupied.update(body)

    return occupied


def _enemy_distance_maps(board: Dict, my_id: str, blocked: Set[Point]) -> List[Dict[Point, int]]:
    width, height = board["width"], board["height"]
    maps = []
    for snake in board.get("snakes", []):
        if snake["id"] == my_id:
            continue
        head = _point(snake["head"])
        maps.append(_bfs_dist([head], blocked - {head}, width, height))
    return maps


def _bfs_dist(sources: Iterable[Point], blocked: Set[Point], width: int, height: int) -> Dict[Point, int]:
    """Shortest free-cell distances from source cells."""
    dist: Dict[Point, int] = {}
    dq: deque[Point] = deque()

    for source in sources:
        if not _in_bounds(source, width, height):
            continue
        if source in blocked:
            continue
        if source not in dist:
            dist[source] = 0
            dq.append(source)

    while dq:
        current = dq.popleft()
        for delta in DIRECTIONS.values():
            nb = _add(current, delta)
            if not _in_bounds(nb, width, height):
                continue
            if nb in blocked or nb in dist:
                continue
            dist[nb] = dist[current] + 1
            dq.append(nb)

    return dist


def _flood_fill(start: Point, blocked: Set[Point], width: int, height: int, limit: int) -> int:
    """Count open cells reachable from start, capped at limit."""
    if not _in_bounds(start, width, height):
        return 0
    if start in blocked:
        return 0

    seen: Set[Point] = {start}
    stack: List[Point] = [start]
    count = 0

    while stack and count < limit:
        cell = stack.pop()
        count += 1
        for delta in DIRECTIONS.values():
            nb = _add(cell, delta)
            if nb in seen:
                continue
            if not _in_bounds(nb, width, height):
                continue
            if nb in blocked:
                continue
            seen.add(nb)
            stack.append(nb)

    return count


def _reachable(start: Point, target: Point, blocked: Set[Point], width: int, height: int) -> bool:
    if start == target:
        return True
    return target in _bfs_dist([start], blocked, width, height)


def _count_escape_routes(cell: Point, blocked: Set[Point], width: int, height: int) -> int:
    count = 0
    for delta in DIRECTIONS.values():
        nb = _add(cell, delta)
        if _in_bounds(nb, width, height) and nb not in blocked:
            count += 1
    return count


def _emergency_move(game_state: Dict) -> str:
    """Last-resort move: any in-bounds non-body move, else up."""
    try:
        board = game_state["board"]
        you = game_state["you"]
        width, height = board["width"], board["height"]
        head = _point(you["head"])
        occupied = _occupied_cells(board.get("snakes", []), include_tails=False)

        for move in ["up", "right", "down", "left"]:
            nxt = _add(head, DIRECTIONS[move])
            if _in_bounds(nxt, width, height) and nxt not in occupied:
                return move
    except Exception:  # noqa: BLE001
        pass
    return "up"


def _point(obj: Dict[str, int]) -> Point:
    return int(obj["x"]), int(obj["y"])


def _add(a: Point, b: Point) -> Point:
    return a[0] + b[0], a[1] + b[1]


def _in_bounds(p: Point, width: int, height: int) -> bool:
    return 0 <= p[0] < width and 0 <= p[1] < height