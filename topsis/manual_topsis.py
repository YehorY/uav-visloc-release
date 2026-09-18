import numpy as np


def apply_topsis(decision_matrix: np.ndarray, weights: np.ndarray, criteria_types: list[str]) -> tuple[np.ndarray, int]:
    """
    Применяет метод TOPSIS к матрице решений.

    Args:
        decision_matrix (np.ndarray): Матрица (альтернативы x критерии).
        weights (np.ndarray): Веса критериев.
        criteria_types (list[str]): Список типов критериев ('cost' или 'benefit').
    """
    if decision_matrix.shape[1] != len(weights) or decision_matrix.shape[1] != len(criteria_types):
        raise ValueError("Размеры матрицы, весов и типов критериев не совпадают")

    # 1. Нормализация
    norm_matrix = decision_matrix / np.sqrt(np.sum(decision_matrix ** 2, axis=0))

    # 2. Взвешивание
    weighted_matrix = norm_matrix * weights

    # 3. Определение идеальных и анти-идеальных решений
    ideal_solution = np.zeros(len(weights))
    negative_ideal_solution = np.zeros(len(weights))

    for i, criteria_type in enumerate(criteria_types):
        if criteria_type == 'cost':  # Чем меньше, тем лучше
            ideal_solution[i] = np.min(weighted_matrix[:, i])
            negative_ideal_solution[i] = np.max(weighted_matrix[:, i])
        elif criteria_type == 'benefit':  # Чем больше, тем лучше
            ideal_solution[i] = np.max(weighted_matrix[:, i])
            negative_ideal_solution[i] = np.min(weighted_matrix[:, i])

    # 4. Расчет расстояний
    dist_to_ideal = np.sqrt(np.sum((weighted_matrix - ideal_solution) ** 2, axis=1))
    dist_to_negative_ideal = np.sqrt(np.sum((weighted_matrix - negative_ideal_solution) ** 2, axis=1))

    # 5. Расчет близости к идеальному решению
    with np.errstate(divide='ignore', invalid='ignore'):
        performance_score = dist_to_negative_ideal / (dist_to_ideal + dist_to_negative_ideal)
        performance_score[np.isnan(performance_score)] = 0

    # 6. Ранжирование
    best_alternative_index = np.argmax(performance_score)

    return performance_score, best_alternative_index