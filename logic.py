"""Логика выбора хода для Battlesnake.

Приоритет стратегий:
  1) choose_move_lookahead — 2-ply paranoid поиск с учётом еды, голода и длины;
  2) choose_move_model     — линейная ранжирующая модель (резерв);
  3) choose_move_heuristic — компактная эвристика (последний резерв).

Система координат: (0, 0) — нижний левый угол.
  up -> y + 1, down -> y - 1, left -> x - 1, right -> x + 1
Схема состояния игры: https://docs.battlesnake.com/api
"""

import time
from collections import deque
from itertools import product
from typing import Dict, List, Optional, Set, Tuple

Point = Tuple[int, int]

DIRECTIONS: Dict[str, Point] = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}

LOOKAHEAD_TIMEOUT_MS = 380      # у Battlesnake лимит 500ms, оставляем буфер
HEAD_TO_HEAD_PENALTY = 10_000   # штраф за потенциально проигранное лобовое столкновение
HUNGRY_THRESHOLD = 50           # ниже этого здоровья активно ищем еду

_BIG = 10_000
_NEIGHBORS = ((0, 1), (0, -1), (-1, 0), (1, 0))


# --- Базовые геометрические утилиты -----------------------------------------

def _in_bounds(p: Point, width: int, height: int) -> bool:
    return 0 <= p[0] < width and 0 <= p[1] < height


def _manhattan(a: Point, b: Point) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _occupied_cells(snakes: List[Dict]) -> Set[Point]:
    """Все клетки, занятые телами змеек прямо сейчас (хвосты включены)."""
    occupied: Set[Point] = set()
    for snake in snakes:
        for seg in snake["body"]:
            occupied.add((seg["x"], seg["y"]))
    return occupied


def _occupied_cells_next_turn(snakes: List[Dict]) -> Set[Point]:
    """Клетки, занятые ПОСЛЕ хода: хвост освобождается, если змейка не ела."""
    occupied: Set[Point] = set()
    for snake in snakes:
        body = snake["body"]
        just_ate = (
            len(body) >= 2
            and body[-1]["x"] == body[-2]["x"]
            and body[-1]["y"] == body[-2]["y"]
        )
        end = len(body) if just_ate else len(body) - 1
        for seg in body[:end]:
            occupied.add((seg["x"], seg["y"]))
    return occupied


def _head_to_head_cells(snakes: List[Dict], my_id: str, my_length: int) -> Set[Point]:
    """Клетки у голов врагов, длина которых >= нашей (лобовое столкновение опасно)."""
    danger: Set[Point] = set()
    for snake in snakes:
        if snake["id"] == my_id or snake["length"] < my_length:
            continue
        ex, ey = snake["head"]["x"], snake["head"]["y"]
        for dx, dy in DIRECTIONS.values():
            danger.add((ex + dx, ey + dy))
    return danger


def _killable_cells(snakes: List[Dict], my_id: str, my_length: int) -> Set[Point]:
    """Клетки, где можно выиграть лобовое столкновение (мы строго длиннее)."""
    kills: Set[Point] = set()
    for snake in snakes:
        if snake["id"] == my_id or snake["length"] >= my_length:
            continue
        ex, ey = snake["head"]["x"], snake["head"]["y"]
        for dx, dy in DIRECTIONS.values():
            kills.add((ex + dx, ey + dy))
    return kills


def _flood_fill(start: Point, occupied: Set[Point], width: int, height: int, limit: int) -> int:
    """Количество достижимых свободных клеток из start (ограничено limit)."""
    seen: Set[Point] = {start}
    stack: List[Point] = [start]
    count = 0
    while stack:
        x, y = stack.pop()
        count += 1
        if count >= limit:
            break
        for dx, dy in _NEIGHBORS:
            nbr = (x + dx, y + dy)
            if nbr in seen or not _in_bounds(nbr, width, height) or nbr in occupied:
                continue
            seen.add(nbr)
            stack.append(nbr)
    return count


def _bfs_dist(sources, blocked, width, height) -> Dict[Point, int]:
    """Кратчайшие расстояния по свободным клеткам от набора источников."""
    dist: Dict[Point, int] = {}
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


# --- Симуляция и оценка (lookahead) -----------------------------------------

def _simulate_one_move(snakes: List[Dict], moves: Dict[str, str], width: int,
                       height: int, foods: Optional[Set[Point]] = None) -> List[Dict]:
    """Симулирует один ход всех змеек с учётом еды и голода.

    Возвращает список выживших змеек (в их состоянии после хода).
    """
    foods = foods if foods is not None else set()

    new_heads: Dict[str, Point] = {}
    for snake in snakes:
        sid = snake["id"]
        if sid not in moves:
            continue
        dx, dy = DIRECTIONS[moves[sid]]
        head = snake["head"]
        new_heads[sid] = (head["x"] + dx, head["y"] + dy)

    length_by_id = {s["id"]: s["length"] for s in snakes}

    # Лобовые столкновения голов.
    dead: Set[str] = set()
    items = list(new_heads.items())
    for i in range(len(items)):
        id1, h1 = items[i]
        for j in range(i + 1, len(items)):
            id2, h2 = items[j]
            if h1 == h2:
                l1, l2 = length_by_id[id1], length_by_id[id2]
                if l1 <= l2:
                    dead.add(id1)
                if l2 <= l1:
                    dead.add(id2)

    # Движение + еда + голод + выход за границы.
    moved: List[Dict] = []
    for snake in snakes:
        sid = snake["id"]
        if sid not in moves or sid in dead:
            continue
        nh = new_heads[sid]
        if not _in_bounds(nh, width, height):
            continue

        ate = nh in foods
        if ate:
            health = 100
            new_body = [{"x": nh[0], "y": nh[1]}] + snake["body"]
            new_length = snake["length"] + 1
        else:
            health = snake["health"] - 1
            new_body = [{"x": nh[0], "y": nh[1]}] + snake["body"][:-1]
            new_length = snake["length"]

        if health <= 0:
            continue  # смерть от голода

        moved.append({
            "id": sid,
            "head": {"x": nh[0], "y": nh[1]},
            "body": new_body,
            "length": new_length,
            "health": health,
        })

    # Столкновения с телами (включая своё).
    all_bodies: Set[Point] = set()
    for s in moved:
        for seg in s["body"][1:]:
            all_bodies.add((seg["x"], seg["y"]))

    return [s for s in moved if (s["head"]["x"], s["head"]["y"]) not in all_bodies]


def _evaluate_state_quick(game_state: Dict, my_id: str) -> float:
    """Оценка листового узла: Voronoi + пространство + длина + притяжение к еде."""
    board = game_state["board"]
    width, height = board["width"], board["height"]
    snakes = board["snakes"]

    me = next((s for s in snakes if s["id"] == my_id), None)
    if me is None:
        return -1_000_000.0          # мы мертвы
    if len(snakes) == 1:
        return 1_000_000.0           # единственные выжившие

    # Змейки уже сдвинуты симуляцией — используем текущую окупацию.
    occupied = _occupied_cells(snakes)
    head = (me["head"]["x"], me["head"]["y"])
    my_length = me["length"]
    health = me["health"]

    enemy_heads = [(s["head"]["x"], s["head"]["y"]) for s in snakes if s["id"] != my_id]
    max_enemy_len = max((s["length"] for s in snakes if s["id"] != my_id), default=0)

    my_dist = _bfs_dist([head], occupied, width, height)
    enemy_dist = _bfs_dist(enemy_heads, occupied, width, height) if enemy_heads else {}

    voronoi = sum(1 for cell, md in my_dist.items() if md < enemy_dist.get(cell, _BIG))
    my_space = _flood_fill(head, occupied, width, height, limit=width * height)

    score = voronoi * 2.0 + my_space * 0.5
    score += (my_length - max_enemy_len) * 15.0   # преимущество в длине

    # Притяжение к ближайшей достижимой еде.
    foods = [(f["x"], f["y"]) for f in board["food"]]
    if foods:
        nearest = min((my_dist.get(f, _BIG) for f in foods))
        if nearest < _BIG:
            health_urgency = max(0.0, (100.0 - health) / 100.0)
            length_gap = max_enemy_len - my_length
            if length_gap >= 0:
                growth_desire = 3.0 + length_gap          # отстаём/равны — едим активно
            else:
                growth_desire = max(0.5, 1.5 + length_gap * 0.4)  # длиннее — умереннее
            weight = 0.5 + growth_desire + health_urgency * 6.0
            score += (width + height - nearest) * weight

    return score


def choose_move_lookahead(game_state: Dict) -> Optional[str]:
    """2-ply paranoid lookahead: максимизируем наш счёт при худшем ответе врагов."""
    start_time = time.time()
    board = game_state["board"]
    width, height = board["width"], board["height"]
    my_id = game_state["you"]["id"]

    legal = _legal_moves(game_state)
    if not legal:
        return None

    snakes = board["snakes"]
    foods = {(f["x"], f["y"]) for f in board["food"]}
    enemies = [s for s in snakes if s["id"] != my_id]
    my_head = (game_state["you"]["head"]["x"], game_state["you"]["head"]["y"])

    best_move, best_score = None, float("-inf")

    for my_move in legal:
        if (time.time() - start_time) * 1000 > LOOKAHEAD_TIMEOUT_MS:
            break

        if not enemies:
            # Соло-режим — просто оцениваем результат хода.
            new_snakes = _simulate_one_move(snakes, {my_id: my_move}, width, height, foods)
            if not any(s["id"] == my_id for s in new_snakes):
                continue
            score = _evaluate_state_quick(
                {**game_state, "board": {**board, "snakes": new_snakes}}, my_id
            )
        else:
            occ = _occupied_cells(snakes)

            # Легальные ходы врагов + расстояние до нас (для приоритизации).
            options = []
            for enemy in enemies:
                eh = (enemy["head"]["x"], enemy["head"]["y"])
                e_legal = [
                    m
                    for m, (dx, dy) in DIRECTIONS.items()
                    if _in_bounds((eh[0] + dx, eh[1] + dy), width, height)
                    and (eh[0] + dx, eh[1] + dy) not in occ
                ] or ["up"]
                options.append((enemy["id"], e_legal, _manhattan(eh, my_head)))

            # Полностью раскрываем ближайших врагов в рамках бюджета (<=16 комбинаций),
            # дальним фиксируем один правдоподобный ход.
            options.sort(key=lambda t: t[2])
            expand: List[Tuple[str, List[str]]] = []
            fixed: Dict[str, str] = {}
            size = 1
            for eid, elegal, _ in options:
                if size * len(elegal) <= 16:
                    expand.append((eid, elegal))
                    size *= len(elegal)
                else:
                    fixed[eid] = elegal[0]

            min_score = float("inf")
            for combo in product(*[opts for _, opts in expand]):
                moves = {my_id: my_move, **fixed}
                for (eid, _), emove in zip(expand, combo):
                    moves[eid] = emove
                new_snakes = _simulate_one_move(snakes, moves, width, height, foods)
                s = _evaluate_state_quick(
                    {**game_state, "board": {**board, "snakes": new_snakes}}, my_id
                )
                min_score = min(min_score, s)
            score = min_score

        if score > best_score:
            best_score, best_move = score, my_move

    return best_move


# --- Резерв 2: линейная модель ----------------------------------------------

def _hunger_priority(you: Dict, snakes: List[Dict]) -> float:
    """Приоритет еды от 0.0 до 1.0 с учётом здоровья и относительной длины."""
    health = you["health"]
    my_length = you["length"]
    enemies = [s for s in snakes if s["id"] != you["id"]]

    if not enemies:
        return 1.0 if health < 40 else 0.0

    max_enemy_length = max(s["length"] for s in enemies)
    if max_enemy_length >= my_length:
        base_threshold = 80
    elif max_enemy_length >= my_length - 2:
        base_threshold = 60
    else:
        base_threshold = 35

    if health >= base_threshold:
        return 0.0
    return (base_threshold - health) / base_threshold


def _safe_foods(foods, my_head, occupied, width, height, my_length, snakes, my_id):
    """Список (еда, приоритет) только для безопасно достижимой еды."""
    result = []
    my_dist_map = _bfs_dist([my_head], occupied, width, height)

    for food in foods:
        my_d = my_dist_map.get(food, _BIG)
        if my_d == _BIG:
            continue

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

        if min_enemy_d <= my_d and closest_enemy_length >= my_length:
            continue  # опасная еда — враг успеет раньше и не короче нас

        growth_value = 1.0
        enemies = [s for s in snakes if s["id"] != my_id]
        if enemies:
            max_enemy_len = max(s["length"] for s in enemies)
            if my_length <= max_enemy_len:
                growth_value = 3.0
            elif my_length == max_enemy_len + 1:
                growth_value = 1.5

        result.append((food, growth_value / (my_d + 1)))

    return sorted(result, key=lambda x: -x[1])


def _candidate_features(state: Dict, move: str) -> Dict[str, float]:
    """Вектор признаков для хода move (move считается легальным)."""
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

    my_dist = _bfs_dist([nxt], occupied, width, height)
    enemy_dist = _bfs_dist(enemy_heads, occupied, width, height) if enemy_heads else {}
    voronoi = sum(1 for cell, md in my_dist.items() if md < enemy_dist.get(cell, _BIG))

    my_tail = (you["body"][-1]["x"], you["body"][-1]["y"])
    reach = _bfs_dist([nxt], occupied - {my_tail}, width, height)
    reaches_tail = 1.0 if my_tail in reach else 0.0

    length_deficit = float(max(0, max_enemy_length - my_length + 1))

    safe_food_nearby = 0.0
    hunger_priority = _hunger_priority(you, board["snakes"])
    if foods:
        safe_list = _safe_foods(
            foods, nxt, occupied, width, height, my_length, board["snakes"], you["id"]
        )
        if safe_list:
            best_food, _ = safe_list[0]
            safe_food_nearby = float(width + height - _manhattan(nxt, best_food))

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


_MODEL: Dict = {
    "feature_names": [
        "space_capped", "open_space", "voronoi", "reaches_tail", "escape",
        "h2h_danger", "near_bigger_head", "near_enemy_head", "wall_dist",
        "food_score", "food_delta", "is_food", "dist_to_center",
    ],
    "mean": [
        7.357954545454546, 100.9034090909091, 48.26988636363637, 0.9943181818181818,
        2.4431818181818183, 0.04261363636363636, 9.673295454545455, 4.676136363636363,
        1.625, 0.8920454545454546, 0.14772727272727273, 0.036931818181818184,
        5.056818181818182,
    ],
    "std": [
        3.5995966185276513, 22.80542174802676, 31.41119158524981, 0.07516338951888041,
        0.6235520417417705, 0.20198444088469822, 7.9675173248507924, 2.2532045017839604,
        1.3552297691803878, 5.861056404757769, 0.9449599886584031, 0.18859442989548575,
        2.34451950177747,
    ],
    "coef": [
        0.00010539398521136327, -1.6778512168946185, 80.89420182766183, 9.793855564450467,
        0.7884630868036275, -11.025170822665032, -0.7981723553489, 0.5410534990053248,
        1.5629078731518526, 7.582325762611304, 0.12463070008097832, 0.21036618806863483,
        1.836259515524985,
    ],
    "intercept": 0.0,
    "top1_accuracy": 0.9928571428571429,
}


def choose_move_model(game_state: Dict) -> Optional[str]:
    """Оценивает каждый легальный ход моделью и возвращает лучший."""
    legal = _legal_moves(game_state)
    if not legal:
        return None

    names = _MODEL["feature_names"]
    mean, std = _MODEL["mean"], _MODEL["std"]
    coef, intercept = _MODEL["coef"], _MODEL["intercept"]

    board = game_state["board"]
    you = game_state["you"]
    width, height = board["width"], board["height"]
    my_length = you["length"]
    occupied = _occupied_cells(board["snakes"])
    head_pt = (you["head"]["x"], you["head"]["y"])

    scores: Dict[str, float] = {}
    for move in legal:
        feats = _candidate_features(game_state, move)
        score = intercept
        for i, name in enumerate(names):
            z = (feats.get(name, 0.0) - mean[i]) / std[i] if std[i] else 0.0
            score += coef[i] * z

        # Ручные поправки поверх модели.
        safe_fn = feats.get("safe_food_nearby", 0.0)
        if safe_fn > 0:
            max_enemy_len = max(
                (s["length"] for s in board["snakes"] if s["id"] != you["id"]),
                default=0,
            )
            length_factor = max(1.0, 1.0 + (max_enemy_len - my_length) * 0.5)
            score += safe_fn * length_factor * 1.8   # проактивный поиск еды

        hunger_p = _hunger_priority(you, board["snakes"])
        if hunger_p > 0:
            score += safe_fn * hunger_p * 15.0        # экстренный голод

        if feats.get("can_kill", 0.0) > 0:
            score += 30.0 * (1.0 - min(1.0, hunger_p * 2))  # убийство слабых, когда сыты

        dx, dy = DIRECTIONS[move]
        nxt = (head_pt[0] + dx, head_pt[1] + dy)
        if _flood_fill(nxt, occupied, width, height, limit=width * height) < my_length:
            score -= 50_000                            # хард-блок ловушки

        scores[move] = score

    return max(scores, key=scores.__getitem__)


# --- Резерв 3: компактная эвристика -----------------------------------------

def choose_move_heuristic(game_state: Dict) -> str:
    """Простая безопасная эвристика: свободное пространство + притяжение к еде."""
    board = game_state["board"]
    you = game_state["you"]
    width, height = board["width"], board["height"]

    head = (you["head"]["x"], you["head"]["y"])
    my_length = you["length"]
    health = you["health"]

    occupied = _occupied_cells(board["snakes"])
    danger = _head_to_head_cells(board["snakes"], you["id"], my_length)
    foods = [(f["x"], f["y"]) for f in board["food"]]

    best_move, best_score = None, float("-inf")
    for move, (dx, dy) in DIRECTIONS.items():
        nxt = (head[0] + dx, head[1] + dy)
        if not _in_bounds(nxt, width, height) or nxt in occupied:
            continue

        score = float(_flood_fill(nxt, occupied, width, height, limit=my_length + 1))
        if nxt in danger:
            score -= HEAD_TO_HEAD_PENALTY
        if foods and health < HUNGRY_THRESHOLD:
            nearest = min(_manhattan(nxt, f) for f in foods)
            score += (width + height - nearest) * 2

        if score > best_score:
            best_score, best_move = score, move

    return best_move or "up"


# --- Точка входа + метаданные -----------------------------------------------

def choose_move(game_state: Dict) -> str:
    """Возвращает ход: lookahead -> модель -> эвристика."""
    try:
        move = choose_move_lookahead(game_state)
        if move:
            return move
    except Exception:  # noqa: BLE001 — сбой стратегии не должен ронять игру
        pass

    try:
        move = choose_move_model(game_state)
        if move:
            return move
    except Exception:  # noqa: BLE001
        pass

    return choose_move_heuristic(game_state)


def get_info() -> Dict[str, str]:
    """Внешний вид и метаданные для GET /."""
    return {
        "apiversion": "1",
        "author": "hackathon",
        "color": "#6434eb",
        "head": "smart-caterpillar",
        "tail": "weight",
        "version": "0.2.0",
    }