"""Model-backed move-selection logic for the Battlesnake.

The served policy uses a linear ranking model, scores each legal move,
and returns the highest-scoring direction. A compact heuristic remains as a
fallback so gameplay still returns a legal move if model scoring fails.

Board coordinates: ``(0, 0)`` is the bottom-left corner.
  up    -> y + 1
  down  -> y - 1
  left  -> x - 1
  right -> x + 1

Game-state schema reference: https://docs.battlesnake.com/api
"""

# new
import time

LOOKAHEAD_TIMEOUT_MS = 380  # у Battlesnake 500ms лимит, оставляем буфер

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

Point = Tuple[int, int]

DIRECTIONS: Dict[str, Point] = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}

# Penalty applied to a move that could lose a head-to-head collision.
HEAD_TO_HEAD_PENALTY = 10_000
# Below this health we start actively steering toward food.
HUNGRY_THRESHOLD = 50

def _simulate_one_move(snakes: List[Dict], moves: Dict[str, str], width: int, height: int) -> List[Dict]:
    """Симулирует один ход всех змеек. Возвращает новый список змеек или пустой список если коллизия."""
    new_heads = {}
    for snake in snakes:
        if snake["id"] not in moves:
            continue
        move = moves[snake["id"]]
        dx, dy = DIRECTIONS[move]
        head = snake["head"]
        new_heads[snake["id"]] = (head["x"] + dx, head["y"] + dy)
    
    # Определяем коллизии голов
    dead = set()
    head_list = list(new_heads.items())
    for i, (id1, h1) in enumerate(head_list):
        for j, (id2, h2) in enumerate(head_list):
            if i >= j:
                continue
            if h1 == h2:
                # Меньшая или равная по длине умирает
                len1 = next(s["length"] for s in snakes if s["id"] == id1)
                len2 = next(s["length"] for s in snakes if s["id"] == id2)
                if len1 <= len2:
                    dead.add(id1)
                if len2 <= len1:
                    dead.add(id2)
    
    new_snakes = []
    foods = set()  # для простоты не симулируем еду в lookahead
    for snake in snakes:
        if snake["id"] not in moves or snake["id"] in dead:
            continue
        nh = new_heads[snake["id"]]
        if not (0 <= nh[0] < width and 0 <= nh[1] < height):
            continue  # вылетел за борт — умер
        
        new_body = [{"x": nh[0], "y": nh[1]}] + snake["body"][:-1]
        new_snakes.append({
            "id": snake["id"],
            "head": {"x": nh[0], "y": nh[1]},
            "body": new_body,
            "length": snake["length"],
            "health": snake["health"] - 1,
        })
    
    # Проверяем коллизии с телами
    all_bodies = set()
    for s in new_snakes:
        for seg in s["body"][1:]:  # исключаем голову
            all_bodies.add((seg["x"], seg["y"]))
    
    survivors = []
    for snake in new_snakes:
        hx, hy = snake["head"]["x"], snake["head"]["y"]
        if (hx, hy) not in all_bodies:
            survivors.append(snake)
    
    return survivors

def _evaluate_state_quick(game_state: Dict, my_id: str) -> float:
    """Быстрая оценка состояния для листовых узлов lookahead."""
    board = game_state["board"]
    width, height = board["width"], board["height"]
    snakes = board["snakes"]
    
    me = next((s for s in snakes if s["id"] == my_id), None)
    if me is None:
        return -100_000  # мы мертвы
    
    if len(snakes) == 1:
        return 100_000  # единственная выжившая
    
    occupied = _occupied_cells_next_turn(snakes)
    head = (me["head"]["x"], me["head"]["y"])
    
    # Воронои
    enemy_heads = [(s["head"]["x"], s["head"]["y"]) for s in snakes if s["id"] != my_id]
    my_dist = _bfs_dist([head], occupied, width, height)
    enemy_dist = _bfs_dist(enemy_heads, occupied, width, height) if enemy_heads else {}
    voronoi = sum(1 for cell, md in my_dist.items() if md < enemy_dist.get(cell, _BIG))
    
    my_space = _flood_fill(head, occupied, width, height, limit=width * height)
    
    return float(voronoi) * 2.0 + float(my_space) * 0.5

def choose_move_lookahead(game_state: Dict) -> Optional[str]:
    """2-ply paranoid lookahead: максимизируем наш счёт при худшем ходе врагов."""
    start_time = time.time()
    board = game_state["board"]
    width, height = board["width"], board["height"]
    my_id = game_state["you"]["id"]
    
    legal = _legal_moves(game_state)
    if not legal:
        return None
    
    snakes = board["snakes"]
    enemies = [s for s in snakes if s["id"] != my_id]
    
    best_move, best_score = None, float("-inf")
    
    for my_move in legal:
        if (time.time() - start_time) * 1000 > LOOKAHEAD_TIMEOUT_MS:
            break
        
        if not enemies:
            # Соло режим — просто оцениваем
            moves = {my_id: my_move}
            new_snakes = _simulate_one_move(snakes, moves, width, height)
            if not any(s["id"] == my_id for s in new_snakes):
                continue
            new_state = {**game_state, "board": {**board, "snakes": new_snakes}}
            score = _evaluate_state_quick(new_state, my_id)
        else:
            # Paranoid: берём минимум по всем враждебным ходам
            min_score = float("inf")
            
            # Для каждого врага берём его лучший ход против нас
            enemy_move_options = []
            for enemy in enemies:
                e_legal = [
                    m for m, (dx, dy) in DIRECTIONS.items()
                    if _in_bounds((enemy["head"]["x"] + dx, enemy["head"]["y"] + dy), width, height)
                    and (enemy["head"]["x"] + dx, enemy["head"]["y"] + dy) not in _occupied_cells(snakes)
                ]
                enemy_move_options.append((enemy["id"], e_legal or ["up"]))
            
            # Перебираем комбинации ходов врагов (ограничиваем для скорости)
            from itertools import product
            enemy_combos = list(product(*[opts for _, opts in enemy_move_options]))
            if len(enemy_combos) > 16:
                enemy_combos = enemy_combos[:16]  # ограничиваем взрыв комбинаций
            
            for combo in enemy_combos:
                moves = {my_id: my_move}
                for (eid, _), emove in zip(enemy_move_options, combo):
                    moves[eid] = emove
                
                new_snakes = _simulate_one_move(snakes, moves, width, height)
                new_state = {**game_state, "board": {**board, "snakes": new_snakes}}
                score = _evaluate_state_quick(new_state, my_id)
                min_score = min(min_score, score)
            
            score = min_score
        
        if score > best_score:
            best_score, best_move = score, best_move or my_move
            best_move = my_move
    
    return best_move


def _hunger_priority(you: Dict, snakes: List[Dict]) -> float:
    """
    Возвращает коэффициент приоритета еды от 0.0 до 1.0.
    Учитывает здоровье И относительную длину среди врагов.
    """
    health = you["health"]
    my_length = you["length"]
    
    enemies = [s for s in snakes if s["id"] != you["id"]]
    if not enemies:
        # Соло — едим при здоровье < 40
        return 1.0 if health < 40 else 0.0
    
    max_enemy_length = max(s["length"] for s in enemies)
    
    # Если кто-то длиннее нас — едим агрессивно всегда
    if max_enemy_length >= my_length:
        base_threshold = 80  # едим при здоровье < 80
    elif max_enemy_length >= my_length - 2:
        base_threshold = 60  # почти равны — умеренно агрессивны
    else:
        base_threshold = 35  # мы явно длиннее — едим только при нужде
    
    if health >= base_threshold:
        return 0.0
    
    # Плавный коэффициент: чем ниже здоровье, тем выше приоритет
    return (base_threshold - health) / base_threshold

def _safe_foods(
    foods: List[Point],
    my_head: Point,
    enemy_heads: List[Point],
    occupied: Set[Point],
    width: int,
    height: int,
    my_length: int,
    snakes: List[Dict],
    my_id: str,
) -> List[Tuple[Point, float]]:
    """
    Возвращает список (еда, приоритет) только для безопасно достижимой еды.
    Приоритет выше если: мы ближе к еде, еда изолирована от врагов,
    поедание даст нам превосходство в длине.
    """
    result = []
    my_dist_map = _bfs_dist([my_head], occupied, width, height)
    
    for food in foods:
        my_d = my_dist_map.get(food, _BIG)
        if my_d == _BIG:
            continue  # недостижима
        
        # Минимальное расстояние врага до этой еды
        min_enemy_d = _BIG
        closest_enemy_length = 0
        for snake in snakes:
            if snake["id"] == my_id:
                continue
            eh = (snake["head"]["x"], snake["head"]["y"])
            ed = _bfs_dist([eh], occupied, width, height).get(food, _BIG)
            if ed < min_enemy_d:
                min_enemy_d = ed
                closest_enemy_length = snake["length"]
        
        # Еда небезопасна если враг доберётся туда раньше или одновременно
        # и при этом он не короче нас (иначе мы его можем убить)
        if min_enemy_d <= my_d and closest_enemy_length >= my_length:
            continue  # пропускаем опасную еду
        
        # Приоритет: близость + ценность роста
        growth_value = 1.0
        enemies = [s for s in snakes if s["id"] != my_id]
        if enemies:
            max_enemy_len = max(s["length"] for s in enemies)
            if my_length <= max_enemy_len:
                # Нам особенно нужна эта еда — растём быстрее
                growth_value = 3.0
            elif my_length == max_enemy_len + 1:
                growth_value = 1.5
        
        priority = growth_value / (my_d + 1)
        result.append((food, priority))
    
    return sorted(result, key=lambda x: -x[1])


def get_info() -> Dict[str, str]:
    """Appearance + metadata returned from ``GET /``."""
    return {
        "apiversion": "1",
        "author": "hackathon",
        "color": "#6434eb",
        "head": "smart-caterpillar",
        "tail": "weight",
        "version": "0.1.0",
    }


# def choose_move(game_state: Dict) -> str:
#     """Return the next move using the model, with a heuristic fallback."""
#     try:
#         move = choose_move_model(game_state)
#     except Exception:  # noqa: BLE001 - a model issue must never break gameplay
#         move = None
#     if move is not None:
#         return move
#     return choose_move_heuristic(game_state)

# new
def choose_move(game_state: Dict) -> str:
    try:
        move = choose_move_lookahead(game_state)
        if move:
            return move
    except Exception:
        pass
    
    try:
        move = choose_move_model(game_state)
        if move:
            return move
    except Exception:
        pass
    
    return choose_move_heuristic(game_state)

# new
def _killable_cells(snakes: List[Dict], my_id: str, my_length: int) -> Set[Point]:
    """Клетки, куда можно выйти нос-к-носу и выиграть (мы строго длиннее)."""
    kills: Set[Point] = set()
    for snake in snakes:
        if snake["id"] == my_id:
            continue
        if snake["length"] >= my_length:  # строго длиннее нужно быть нам
            continue
        ehead = (snake["head"]["x"], snake["head"]["y"])
        for dx, dy in DIRECTIONS.values():
            kills.add((ehead[0] + dx, ehead[1] + dy))
    return kills


def choose_move_heuristic(game_state: Dict) -> str:
    """Return the next move for the current turn."""
    board = game_state["board"]
    you = game_state["you"]
    width: int = board["width"]
    height: int = board["height"]

    head: Point = (you["head"]["x"], you["head"]["y"])
    my_length: int = you["length"]
    health: int = you["health"]

    occupied = _occupied_cells(board["snakes"])
    danger = _head_to_head_cells(board["snakes"], you["id"], my_length)
    foods = [(f["x"], f["y"]) for f in board["food"]]

    best_move = None
    best_score = float("-inf")

    for move, (dx, dy) in DIRECTIONS.items():
        nxt = (head[0] + dx, head[1] + dy)

        if not _in_bounds(nxt, width, height):
            continue
        if nxt in occupied:
            continue

        # Reachable open space from this cell. If we can't fit our own body in
        # the space we'd be moving into, we're about to trap ourselves.
        space = _flood_fill(nxt, occupied, width, height, limit=my_length + 1)
        score = float(space)

        if nxt in danger:
            score -= HEAD_TO_HEAD_PENALTY

        # When hungry, nudge toward the closest food.
        if foods and health < HUNGRY_THRESHOLD:
            nearest = min(_manhattan(nxt, f) for f in foods)
            score += (width + height - nearest) * 2

        if score > best_score:
            best_score = score
            best_move = move

    # No safe move found -> we're cornered. Move up and hope for the best.
    return best_move or "up"

# old
# def _occupied_cells(snakes: List[Dict]) -> Set[Point]:
#     """All cells currently filled by any snake's body.

#     We keep tails occupied too; they only free up *next* turn and treating them
#     as solid is the conservative, safe choice for a base bot.
#     """
#     occupied: Set[Point] = set()
#     for snake in snakes:
#         for seg in snake["body"]:
#             occupied.add((seg["x"], seg["y"]))
#     return occupied

def _occupied_cells(snakes: List[Dict]) -> Set[Point]:
    occupied: Set[Point] = set()
    for snake in snakes:
        body = snake["body"]
        for seg in body:
            occupied.add((seg["x"], seg["y"]))
    return occupied

# new
def _occupied_cells_next_turn(snakes: List[Dict]) -> Set[Point]:
    """Занятые клетки ПОСЛЕ того, как все змейки сделают ход.
    Хвост освобождается, если змейка не ела (body[-1] != body[-2]).
    """
    occupied: Set[Point] = set()
    for snake in snakes:
        body = snake["body"]
        # Определяем, поела ли змейка: если ела — два последних сегмента одинаковы
        just_ate = len(body) >= 2 and body[-1]["x"] == body[-2]["x"] and body[-1]["y"] == body[-2]["y"]
        end = len(body) if just_ate else len(body) - 1
        for seg in body[:end]:
            occupied.add((seg["x"], seg["y"]))
    return occupied


def _head_to_head_cells(snakes: List[Dict], my_id: str, my_length: int) -> Set[Point]:
    """Cells adjacent to enemy heads that are >= our length.

    Moving onto one of these risks a head-to-head collision we would lose or
    tie, so they are heavily penalized (but not forbidden — sometimes it's the
    only move).
    """
    danger: Set[Point] = set()
    for snake in snakes:
        if snake["id"] == my_id:
            continue
        if snake["length"] < my_length:
            continue
        ehead = (snake["head"]["x"], snake["head"]["y"])
        for dx, dy in DIRECTIONS.values():
            danger.add((ehead[0] + dx, ehead[1] + dy))
    return danger


def _flood_fill(start: Point, occupied: Set[Point], width: int, height: int, limit: int) -> int:
    """Count open cells reachable from ``start`` (capped at ``limit``).

    Used to avoid moves that would seal us into a small pocket.
    """
    seen: Set[Point] = {start}
    stack: List[Point] = [start]
    count = 0
    while stack:
        x, y = stack.pop()
        count += 1
        if count >= limit:
            break
        for dx, dy in DIRECTIONS.values():
            nbr = (x + dx, y + dy)
            if nbr in seen:
                continue
            if not _in_bounds(nbr, width, height):
                continue
            if nbr in occupied:
                continue
            seen.add(nbr)
            stack.append(nbr)
    return count


def _in_bounds(p: Point, width: int, height: int) -> bool:
    return 0 <= p[0] < width and 0 <= p[1] < height


def _manhattan(a: Point, b: Point) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# --- Embedded model features -------------------------------------------------

_BIG = 10_000
_NEIGHBORS = ((0, 1), (0, -1), (-1, 0), (1, 0))


def _bfs_dist(sources, blocked, width, height):
    """Shortest free-cell distances from seed cells."""
    dist = {}
    dq = deque()
    for source in sources:
        if source not in dist:
            dist[source] = 0
            dq.append(source)
    while dq:
        x, y = dq.popleft()
        d = dist[(x, y)]
        for dx, dy in _NEIGHBORS:
            nb = (x + dx, y + dy)
            if 0 <= nb[0] < width and 0 <= nb[1] < height and nb not in blocked and nb not in dist:
                dist[nb] = d + 1
                dq.append(nb)
    return dist


def _candidate_features(state: Dict, move: str) -> Dict[str, float]:
    """Feature vector for playing ``move`` from ``state``. Assumes ``move`` is legal."""
    board = state["board"]
    you = state["you"]
    width, height = board["width"], board["height"]
    head = (you["head"]["x"], you["head"]["y"])
    my_length = you["length"]
    health = you["health"]

    dx, dy = DIRECTIONS[move]
    nxt = (head[0] + dx, head[1] + dy)

    occupied = _occupied_cells_next_turn(board["snakes"])
    danger = _head_to_head_cells(board["snakes"], you["id"], my_length)
    killable = _killable_cells(board["snakes"], you["id"], my_length)
    foods = [(f["x"], f["y"]) for f in board["food"]]
    enemies = [s for s in board["snakes"] if s["id"] != you["id"]]
    max_enemy_length = max((s["length"] for s in enemies), default=0)
    enemy_heads = [(s["head"]["x"], s["head"]["y"]) for s in enemies]
    bigger_heads = [(s["head"]["x"], s["head"]["y"]) for s in enemies if s["length"] >= my_length]

    # Voronoi control: cells we reach strictly before any enemy.
    my_dist = _bfs_dist([nxt], occupied, width, height)
    enemy_dist = _bfs_dist(enemy_heads, occupied, width, height) if enemy_heads else {}
    voronoi = sum(1 for cell, md in my_dist.items() if md < enemy_dist.get(cell, _BIG))

    # Tail reachability is a useful anti-self-trap signal.
    my_tail = (you["body"][-1]["x"], you["body"][-1]["y"])
    reach = _bfs_dist([nxt], occupied - {my_tail}, width, height)
    reaches_tail = 1.0 if my_tail in reach else 0.0

    # Насколько мы короче самого длинного врага
    length_deficit = float(max(0, max_enemy_length - my_length + 1))

    # Есть ли безопасная еда рядом (в 3 ходах)
    safe_food_nearby = 0.0
    hunger_priority = _hunger_priority(you, board["snakes"])
    if foods:
        safe_list = _safe_foods(
            foods, nxt, 
            [(s["head"]["x"], s["head"]["y"]) for s in enemies],
            occupied, width, height, my_length, board["snakes"], you["id"]
        )
        if safe_list:
            best_food, _ = safe_list[0]
            dist_to_safe_food = _manhattan(nxt, best_food)
            safe_food_nearby = float(width + height - dist_to_safe_food)

    # Ценность роста: насколько важно сейчас есть
    growth_urgency = length_deficit * hunger_priority

    escape = sum(
        1
        for ddx, ddy in _NEIGHBORS
        if _in_bounds((nxt[0] + ddx, nxt[1] + ddy), width, height)
        and (nxt[0] + ddx, nxt[1] + ddy) not in occupied
    )

    nearest_now = min((_manhattan(head, f) for f in foods), default=_BIG)
    nearest_next = min((_manhattan(nxt, f) for f in foods), default=_BIG)
    hungry = health < HUNGRY_THRESHOLD

    return {
        "space_capped": float(_flood_fill(nxt, occupied, width, height, limit=my_length + 1)),
        "can_kill": 1.0 if nxt in killable else 0.0,
        "open_space": float(_flood_fill(nxt, occupied, width, height, limit=width * height)),
        "voronoi": float(voronoi),
        "reaches_tail": reaches_tail,
        "escape": float(escape),
        "h2h_danger": 1.0 if nxt in danger else 0.0,
        "near_bigger_head": float(min((_manhattan(nxt, h) for h in bigger_heads), default=width + height)),
        "near_enemy_head": float(min((_manhattan(nxt, h) for h in enemy_heads), default=width + height)),
        "wall_dist": float(min(nxt[0], width - 1 - nxt[0], nxt[1], height - 1 - nxt[1])),
        "food_score": float((width + height - nearest_next) * 2) if hungry and foods else 0.0,
        "food_delta": float(nearest_now - nearest_next) if foods else 0.0,
        "is_food": 1.0 if nxt in foods else 0.0,
        "dist_to_center": abs(nxt[0] - (width - 1) / 2) + abs(nxt[1] - (height - 1) / 2),
        "length_deficit": length_deficit,
        "safe_food_nearby": safe_food_nearby,
        "growth_urgency": growth_urgency,
    }


# --- Model -----------------------------------------------------
# Embedded standardized linear model.

_MODEL: Dict = {
    "feature_names": [
        "space_capped",
        "can_kill",
        "open_space",
        "voronoi",
        "reaches_tail",
        "escape",
        "h2h_danger",
        "near_bigger_head",
        "near_enemy_head",
        "wall_dist",
        "food_score",
        "food_delta",
        "is_food",
        "dist_to_center",
    ],
    "mean": [
        7.357954545454546,
        15,
        100.9034090909091,
        48.26988636363637,
        0.9943181818181818,
        2.4431818181818183,
        0.04261363636363636,
        9.673295454545455,
        4.676136363636363,
        1.625,
        0.8920454545454546,
        0.14772727272727273,
        0.036931818181818184,
        5.056818181818182,
    ],
    "std": [
        3.5995966185276513,
        22.80542174802676,
        31.41119158524981,
        0.07516338951888041,
        0.6235520417417705,
        0.20198444088469822,
        7.9675173248507924,
        2.2532045017839604,
        1.3552297691803878,
        5.861056404757769,
        0.9449599886584031,
        0.18859442989548575,
        2.34451950177747,
    ],
    "coef": [
        0.00010539398521136327,
        -1.6778512168946185,
        80.89420182766183,
        9.793855564450467,
        0.7884630868036275,
        -11.025170822665032,
        -0.7981723553489,
        0.5410534990053248,
        1.5629078731518526,
        7.582325762611304,
        0.12463070008097832,
        0.21036618806863483,
        1.836259515524985,
    ],
    "intercept": 0.0,
    "top1_accuracy": 0.9928571428571429,
}


def choose_move_model(game_state: Dict) -> Optional[str]:
    """Score each legal move with the trained model; return the best.

    Returns ``None`` (so the caller falls back to the heuristic) if the model
    isn't available or the snake is trapped with no legal move.
    """
    legal = _legal_moves(game_state)
    if not legal:
        return None

    names = _MODEL["feature_names"]
    mean = _MODEL["mean"]
    std = _MODEL["std"]
    coef = _MODEL["coef"]
    intercept = _MODEL["intercept"]
    
    # new
    board = game_state["board"]
    you = game_state["you"]
    width, height = board["width"], board["height"]
    my_length = you["length"]
    occupied = _occupied_cells(board["snakes"])

    # old
#     best_move, best_score = None, float("-inf")
#     for move in legal:
#         feats = _candidate_features(game_state, move)
#         score = intercept
#         for i, name in enumerate(names):
#             z = (feats.get(name, 0.0) - mean[i]) / std[i] if std[i] else 0.0
#             score += coef[i] * z
#         if score > best_score:
#             best_score, best_move = score, move

    scores = {}
    for move in legal:
        feats = _candidate_features(game_state, move)
        # Основной score из модели
        score = intercept
        for i, name in enumerate(names):
            z = (feats.get(name, 0.0) - mean[i]) / std[i] if std[i] else 0.0
            score += coef[i] * z
        
        
        # new
        # Ручные поправки поверх модели
        hunger_p = _hunger_priority(game_state["you"], board["snakes"])

        # Агрессивный бонус за безопасную еду пропорционально срочности
        score += feats.get("safe_food_nearby", 0.0) * hunger_p * 0.8

        # Штраф за отставание в росте
        score -= feats.get("length_deficit", 0.0) * 3.0

        # Бонус за срочность роста
        score += feats.get("growth_urgency", 0.0) * 5.0
        
        # Хард-блок: если flood fill меньше нашей длины — почти верная ловушка
        dx, dy = DIRECTIONS[move]
        head = (you["head"]["x"], you["head"]["y"])
        nxt = (head[0] + dx, head[1] + dy)
        space = _flood_fill(nxt, occupied, width, height, limit=width * height)
        if space < my_length:
            score -= 50_000  # жёстко штрафуем, не запрещаем (вдруг других нет)
        
        scores[move] = score

    return max(scores, key=scores.__getitem__)
          
#     return best_move


def _legal_moves(game_state: Dict) -> List[str]:
    board = game_state["board"]
    width, height = board["width"], board["height"]
    head = (game_state["you"]["head"]["x"], game_state["you"]["head"]["y"])
    occupied = _occupied_cells(board["snakes"])
    return [
        move
        for move, (dx, dy) in DIRECTIONS.items()
        if _in_bounds((head[0] + dx, head[1] + dy), width, height)
        and (head[0] + dx, head[1] + dy) not in occupied
    ]
