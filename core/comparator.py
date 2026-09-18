import numpy as np
import networkx as nx


def _get_path_properties(route_points: np.ndarray, type_signature: list) -> dict:
    """
    Новая функция для извлечения всех свойств пути для TOPSIS.
    """
    # 1. Проверки
    if len(route_points) < 2:
        return {'edge_ratios': [], 'internal_angles': [], 'type_signature': []}

    # 3. Вектор длин ребер и их отношений
    edge_lengths = []
    for i in range(len(route_points) - 1):
        edge_lengths.append(np.linalg.norm(route_points[i + 1] - route_points[i]))

    total_length = sum(edge_lengths)
    # Защита от деления на ноль, если путь нулевой длины
    edge_ratios = [(edge / total_length) if total_length > 0 else 0 for edge in edge_lengths]

    # 4. Вектор внутренних углов
    internal_angles = []
    if len(route_points) >= 3:
        for i in range(1, len(route_points) - 1):
            vec1 = route_points[i - 1] - route_points[i]
            vec2 = route_points[i + 1] - route_points[i]
            norm_product = (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            if norm_product < 1e-8:
                cosine_angle = 1.0 # Or 0.0, handling the collinear/zero-length case
            else:
                cosine_angle = np.dot(vec1, vec2) / norm_product
            angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
            internal_angles.append(angle)

    return {
        'edge_ratios': edge_ratios,
        'internal_angles': internal_angles,
        'type_signature': type_signature
    }


def compare_graphs(
        photo_route: list,
        map_routes: list,
        photo_features: dict,  # Получаем готовые фичи фото
        map_normalized_points_list: list[np.ndarray],
        map_types_list: list[list]
) -> list:
    """
    Фильтрует маршруты карты и рассчитывает КРИТЕРИИ ОШИБКИ для TOPSIS.
    """
    candidate_routes = []

    # 1. Получаем эталонные векторы
    photo_ratios = photo_features['edge_ratios']
    photo_angles = photo_features['internal_angles']
    photo_types = photo_features['type_signature']

    for i, map_route in enumerate(map_routes):
        # 2. Получаем признаки кандидата
        map_points = map_normalized_points_list[i]
        map_types = map_types_list[i]
        map_features = _get_path_properties(map_points, map_types)
        map_ratios = map_features['edge_ratios']
        map_angles = map_features['internal_angles']

        # 3. Проверяем, что у них одинаковая структура (кол-во ребер и углов)
        if len(photo_ratios) != len(map_ratios) or len(photo_angles) != len(map_angles):
            continue  # Пропускаем, если структура не совпадает

        # --- Расчет 3-х критериев для TOPSIS ---

        # C1: Ошибка формы (Сумма разниц отношений) - Cost
        ratio_error = sum(abs(p - m) for p, m in zip(photo_ratios, map_ratios))

        # C2: Ошибка углов (Сумма разниц углов) - Cost
        angle_error = sum(abs(p - m) for p, m in zip(photo_angles, map_angles))

        # C3: Совпадение типов - Benefit
        matches = sum(1 for a, b in zip(photo_types, map_types) if a == b)
        type_match_score = matches / len(photo_types) if photo_types else 0

        candidate_routes.append({
            'route': map_route,
            'ratio_error': ratio_error,
            'angle_error': angle_error,
            'type_match_score': type_match_score
        })

    print(f"\nНайдено {len(candidate_routes)} потенциальных кандидатов для TOPSIS.")
    return candidate_routes