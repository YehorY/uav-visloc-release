import numpy as np
import time
from heapq import heappush, heappop


# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---

def build_distance_matrix(points: np.ndarray) -> np.ndarray:
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def _get_farthest_point_id(points: np.ndarray, anchor_id: int) -> int:
    dist_matrix = build_distance_matrix(points)
    return np.argmax(dist_matrix[anchor_id])


def _get_angle_degrees(p1: np.ndarray, p2: np.ndarray) -> float:
    return np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))


def _get_path_angles(points: np.ndarray, path: list[int]) -> list[float]:
    """Возвращает список углов для каждого ребра в пути."""
    angles = []
    for i in range(len(path) - 1):
        p1 = points[path[i]]
        p2 = points[path[i + 1]]
        angles.append(_get_angle_degrees(p1, p2))
    return angles


# --- ПОИСК ЭТАЛОНА (Фото) - без изменений ---

def find_photo_path_beam_search(
        photo_nodes_array: np.ndarray,
        ref_photo_classes_array: np.ndarray,
        anchor_id: int,
        max_vertices: int = 15,
        beam_width: int = 3
) -> list[int] | None:
    beam = [(0, [anchor_id])]  # (score, path)
    dist_matrix = build_distance_matrix(photo_nodes_array)
    for _ in range(max_vertices - 1):
        candidates = []
        for score, path in beam:
            last_node = path[-1]
            for next_node in range(len(photo_nodes_array)):
                if next_node not in path:
                    new_score = score + dist_matrix[last_node, next_node]
                    new_path = path + [next_node]
                    heappush(candidates, (new_score, new_path))
        beam = [heappop(candidates) for _ in range(min(beam_width, len(candidates)))]
        if not beam: break
    return min(beam, key=lambda x: x[0])[1] if beam else None


# --- ЭТАП 1: PRE-FILTER (Карта) ---

def pre_filter_nodes_global(
        map_nodes_array: np.ndarray,
        anchor_id: int,
        ref_photo_nodes_array: np.ndarray,
        ref_photo_anchor_id: int,
        sector_width_degrees: float,
        global_radius: float = 1500.0
) -> tuple[set, float]:
    """
    Отбирает узлы на карте, попадающие в глобальный сектор и радиус.
    (Старый фильтр "отсечения меньших" УДАЛЕН)
    """
    print("\n--- Этап 1: Предварительный отбор узлов на карте ---")

    farthest_photo_id = _get_farthest_point_id(ref_photo_nodes_array, ref_photo_anchor_id)
    anchor_pos_photo = ref_photo_nodes_array[ref_photo_anchor_id]
    farthest_pos_photo = ref_photo_nodes_array[farthest_photo_id]
    bisector_angle = _get_angle_degrees(anchor_pos_photo, farthest_pos_photo)

    candidate_node_ids = set()
    anchor_pos_map = map_nodes_array[anchor_id]
    candidate_node_ids.add(anchor_id)

    stats = {"total": 0, "pruned_by_radius": 0, "pruned_by_sector": 0}

    for node_id, node_pos in enumerate(map_nodes_array):
        stats["total"] += 1
        if node_id == anchor_id:
            continue

        # 1. Фильтр по ГЛОБАЛЬНОМУ радиусу от якоря
        distance = np.linalg.norm(node_pos - anchor_pos_map)
        if distance > global_radius:
            stats["pruned_by_radius"] += 1
            continue

        # 2. Фильтр по ГЛОБАЛЬНОМУ сектору от якоря
        angle_to_node = _get_angle_degrees(anchor_pos_map, node_pos)
        angle_diff = (angle_to_node - bisector_angle + 180) % 360 - 180

        if abs(angle_diff) <= sector_width_degrees / 2:
            candidate_node_ids.add(node_id)
        else:
            stats["pruned_by_sector"] += 1

    print(f"--- Статистика Pre-filter ---")
    print(f"Всего узлов на карте: {stats['total']}")
    print(f"Отсечено (радиус > {global_radius}): {stats['pruned_by_radius']}")
    print(f"Отсечено (вне сектора): {stats['pruned_by_sector']}")
    print(f" ИТОГО УЗЛОВ в рабочей области: {len(candidate_node_ids)}")
    print("--- Отбор завершён ---")

    return candidate_node_ids, bisector_angle


# --- ЭТАП 2: A* BEAM SEARCH (Карта) с ДИНАМИЧЕСКИМ ФИЛЬТРОМ ---

def _get_top_distant_nodes(nodes_array: np.ndarray, anchor_id: int, candidate_ids: set, percentage: float = 0.5) -> set:
    """Возвращает 50% самых дальних узлов из набора кандидатов."""
    anchor_pos = nodes_array[anchor_id]
    distances = []
    for node_id in candidate_ids:
        if node_id == anchor_id: continue
        dist = np.linalg.norm(nodes_array[node_id] - anchor_pos)
        distances.append((dist, node_id))

    distances.sort(key=lambda x: x[0], reverse=True)
    count_to_take = int(len(distances) * percentage)
    target_nodes = {node_id for dist, node_id in distances[:count_to_take]}
    print(f"   (Эвристика: выбрано {len(target_nodes)} целевых узлов из {len(distances)})")
    return target_nodes


def find_map_paths_A_star_beam_search(
        map_nodes_array: np.ndarray,
        map_classes_array: np.ndarray,
        anchor_id: int,
        candidate_node_ids: set,
        ref_photo_nodes_array: np.ndarray,
        ref_photo_classes_array: np.ndarray,
        ref_photo_route: list,
        beam_width: int = 100,
        angle_weight: float = 0.5,
        target_percentage: float = 0.5
) -> list[list[int]]:
    """
    Ищет маршруты на карте с помощью A* Beam Search.
    Включает ДИНАМИЧЕСКИЙ ФИЛЬТР КВАДРАНТОВ.
    """
    print("\n--- Этап 2: Запуск A* Beam Search в рабочей области ---")

    target_nodes = _get_top_distant_nodes(map_nodes_array, anchor_id, candidate_node_ids, target_percentage)
    if not target_nodes: return []

    target_angles = _get_path_angles(ref_photo_nodes_array, ref_photo_route)
    required_len = len(ref_photo_route)

    dist_matrix = build_distance_matrix(map_nodes_array)
    beam = [(0, [anchor_id])]  # (score, path)
    start_time = time.perf_counter()

    for i in range(required_len - 1):
        # i - это индекс ребра, которое мы строим (0 = первое ребро)
        target_angle = target_angles[i]

        candidates = []
        for score, path in beam:
            last_node_id = path[-1]
            last_node_pos = map_nodes_array[last_node_id]

            for next_node_id in candidate_node_ids:
                if next_node_id not in path:

                    next_node_pos = map_nodes_array[next_node_id]

                    target_class = ref_photo_classes_array[ref_photo_route[i + 1]]
                    candidate_class = map_classes_array[next_node_id]
                    if target_class != candidate_class:
                        continue # Prune paths that don't match the object "layer"

                    # --- A* Эвристика ---
                    distance_cost = dist_matrix[last_node_id, next_node_id]
                    candidate_angle = _get_angle_degrees(last_node_pos, next_node_pos)
                    angle_error = (candidate_angle - target_angle + 180) % 360 - 180

                    new_score = score + distance_cost + (angle_weight * abs(angle_error))
                    new_path = path + [next_node_id]
                    heappush(candidates, (new_score, new_path))

        beam = [heappop(candidates) for _ in range(min(beam_width, len(candidates)))]
        if not beam:
            print("   (Поиск A* Beam Search прерван, луч опустел)")
            break

    end_time = time.perf_counter()
    print(f"Время поиска: {end_time - start_time:.4f} секунд")
    print(f"Найдено {len(beam)} путей-кандидатов до эвристической фильтрации.")

    final_paths = [path for score, path in beam if path[-1] in target_nodes]

    print(f"ИТОГО НАЙДЕНО ПУТЕЙ (после эвристики): {len(final_paths)}")
    print("--- Поиск завершён ---")

    return final_paths


def pre_filter_by_flight_vector(
    map_nodes_array: np.ndarray,
    current_pos: np.ndarray,
    heading_degrees: float,
    sector_width_degrees: float,
    max_radius: float
) -> set:
    """Filters map nodes strictly by physical flight distance and directional heading."""
    candidate_ids = set()
    for node_id, node_pos in enumerate(map_nodes_array):
        # 1. Distance check
        dist = np.linalg.norm(node_pos - current_pos)
        if dist > max_radius or dist < 1.0: # Skip exact current pos or too far
            continue
            
        # 2. Omni-directional Near Field OR Sector check
        if dist < 150.0:
            candidate_ids.add(node_id)
        else:
            angle_to_node = np.degrees(np.arctan2(node_pos[1] - current_pos[1], node_pos[0] - current_pos[0]))
            angle_diff = (angle_to_node - heading_degrees + 180) % 360 - 180
            if abs(angle_diff) <= (sector_width_degrees / 2):
                candidate_ids.add(node_id)
            
    return candidate_ids