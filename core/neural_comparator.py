import torch
import torch.nn as nn
import numpy as np
import logging

# Настройка логирования
logger = logging.getLogger("NeuralComparator")


class SiameseNetwork(nn.Module):
    """
    Архитектура должна быть 1-в-1 как при обучении!
    """

    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64)
        )

    def forward_one(self, x):
        # Используем reshape, так как входной тензор может быть не contiguous
        x = x.reshape(x.size(0), -1)
        output = self.fc(x)
        return output


class NeuralComparator:
    def __init__(self, model_path, device_type='cpu'):
        self.device = torch.device(device_type)
        self.model = SiameseNetwork().to(self.device)

        try:
            # Загружаем веса. map_location важен для работы на CPU
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()  # Режим предсказания (обязательно!)
            logger.info("NeuralComparator: Модель успешно загружена.")
        except Exception as e:
            logger.error(f"NeuralComparator: Ошибка загрузки модели {model_path}: {e}")
            raise e

    def _normalize_path(self, path):
        """Центрирование и масштабирование пути в диапазон [-1, 1]"""
        path = np.array(path)
        if len(path) == 0: return path
        centered = path - path[0]  # Сдвиг начала в (0,0)
        max_val = np.max(np.abs(centered))
        if max_val > 1e-6:
            centered = centered / max_val
        return centered

    def compare_batch(self, camera_path, map_candidates):
        """
        Сравнивает путь с камеры со списком кандидатов.
        Возвращает список (index, score), отсортированный по схожести (меньше = лучше).
        """
        if not map_candidates:
            return []

        # 1. Подготовка фото-пути
        cam_norm = self._normalize_path(camera_path)
        cam_tensor = torch.from_numpy(cam_norm).float().unsqueeze(0).to(self.device)

        scores = []

        with torch.no_grad():
            # Получаем эмбеддинг фото один раз
            cam_emb = self.model.forward_one(cam_tensor)

            # 2. Проход по кандидатам
            # Для скорости можно объединить кандидатов в один тензор (Batch Processing)
            # Но для наглядности оставим цикл (на 100 кандидатов это мгновенно)
            for idx, map_path in enumerate(map_candidates):
                map_norm = self._normalize_path(map_path)
                map_tensor = torch.from_numpy(map_norm).float().unsqueeze(0).to(self.device)

                map_emb = self.model.forward_one(map_tensor)

                # Евклидово расстояние
                dist = torch.nn.functional.pairwise_distance(cam_emb, map_emb).item()
                scores.append((idx, dist))

        # Сортировка: чем меньше дистанция, тем больше сходство
        scores.sort(key=lambda x: x[1])
        return scores