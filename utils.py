from torch.utils.data import Subset


def limit_dataset_size(dataset, max_samples=None, max_classes=None):
    """
    Ogranicza rozmiar datasetu:
    - max_samples: maksymalna liczba próbek
    - max_classes: maksymalna liczba klas
    """
    if max_classes is not None:
        # Wybierz podzbiór klas
        class_counts = {}
        selected_indices = []
        for idx, (_, label) in enumerate(dataset):
            if label not in class_counts:
                class_counts[label] = 0
            if len(class_counts) <= max_classes and class_counts[label] < (max_samples or float('inf')):
                selected_indices.append(idx)
                class_counts[label] += 1
        return Subset(dataset, selected_indices)

    elif max_samples is not None:
        # Ogranicz tylko liczbę próbek
        return Subset(dataset, range(min(max_samples, len(dataset))))

    return dataset