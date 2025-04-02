import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from ContrastiveLoss import ContrastiveLoss
from SiameseTestDataset import SiameseTestDataset
from SiameseTrainDataset import SiameseTrainDataset
from SiemseCNN import SiameseCNN
from utils import limit_dataset_size

transform = transforms.Compose([
    transforms.Resize((105, 105)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Wbudowany dataset Omniglot w Torchvision
train_dataset = torchvision.datasets.Omniglot(
    root="./data",
    download=True,
    transform=transform,
    background=True  # zestaw treningowy
)

test_dataset = torchvision.datasets.Omniglot(
    root="./data",
    download=True,
    transform=transform,
    background=False  # zestaw testowy
)

# DataLoadery
train_siamese = SiameseTrainDataset(train_dataset)
test_siamese = SiameseTestDataset(test_dataset)

train_siamese = limit_dataset_size(train_siamese, 5000)

train_loader = DataLoader(train_siamese, batch_size=64, shuffle=True)
test_loader = DataLoader(test_siamese, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseCNN().to(device)
criterion = ContrastiveLoss(margin=1.0)
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train(model, train_loader, optimizer, criterion, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for img1, img2, label in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):

            img1, img2, label = img1.to(device), img2.to(device), label.to(device)

            optimizer.zero_grad()
            out1, out2 = model(img1, img2)
            loss = criterion(out1, out2, label)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

'''
def evaluate_few_shot(model, test_loader, n_way=5, k_shot=1):
    model.eval()
    correct = 0
    total = 0

    for _ in range(10):  # 100 losowych zadań few-shot
        # Wybierz n_way klas i k_shot przykładów na klasę
        classes = np.random.choice(test_siamese.classes, n_way, replace=False)
        support_set = []
        query_set = []

        for class_id in classes:
            samples = np.random.choice(
                np.where(test_siamese.labels == class_id)[0],
                k_shot + 1,  # k_shot dla support + 1 dla query
                replace=False
            )
            support_set.extend(samples[:-1])
            query_set.append(samples[-1])

        # Oblicz prototypy klas (średnie embeddingi)
        prototypes = []
        for idx in support_set:
            img, _ = test_dataset[idx]
            img = img.unsqueeze(0).to(device)
            emb = model.forward_one(img)
            prototypes.append(emb)
        prototypes = torch.stack(prototypes).mean(dim=0)

        # Klasyfikuj query
        for q_idx in query_set:
            img, true_label = test_dataset[q_idx]
            img = img.unsqueeze(0).to(device)
            q_emb = model.forward_one(img)

            similarities = F.cosine_similarity(q_emb, prototypes, dim=1)  # Wynik w [-1, 1]
            print(similarities)
            pred = classes[torch.argmax(similarities)]

            # Oblicz odległości do prototypów
            # distances = F.pairwise_distance(q_emb, prototypes)
            # pred = classes[torch.argmin(distances)]

            if pred == true_label:
                correct += 1
            total += 1

    accuracy = correct / total
    print(f"Few-Shot Accuracy ({n_way}-way {k_shot}-shot): {accuracy * 100:.2f}%")
    return accuracy
'''


def evaluate_few_shot(model, test_dataset, n_way=5, k_shot=2, num_queries=5, num_episodes=10, device="cpu"  ):
    model.eval()
    correct = 0
    total = 0

    for _ in range(num_episodes):
        # 1. Losuj n_way klas
        classes = np.random.choice(test_dataset.classes, n_way, replace=False)
        # print(f'Classes: {classes}')
        support_indices = []
        query_indices = []

        # 2. Dla każdej klasy wybierz k_shot supportów i num_queries query
        for cls in classes:
            cls_indices = np.where(test_dataset.labels == cls)[0]
            selected = np.random.choice(cls_indices, k_shot + num_queries, replace=False)
            support_indices.append(selected[:k_shot])
            query_indices.extend(selected[k_shot:])

        # print(f'Support set indices e.x.: {support_indices[0]}, length: {len(support_indices)}')
        # print(f'Query set indices e.x.: {query_indices[0]}, length: {len(query_indices)}')

        # 3. Oblicz prototypy klas (uśrednione embeddingi supportów)
        prototypes = []
        for i in range(n_way):
            class_embeddings = []
            for idx in support_indices[i]:  # Teraz poprawne indeksy
                img, _ = test_dataset[idx]
                img = img.unsqueeze(0).to(device)
                emb = model.forward_one(img).detach()
                class_embeddings.append(emb)
            prototype = torch.mean(torch.stack(class_embeddings), dim=0)
            prototypes.append(prototype)

        # 4. Klasyfikuj query
        for idx in query_indices:
            img, true_label = test_dataset[idx]
            img = img.unsqueeze(0).to(device)  # Przenoszenie na właściwe urządzenie
            q_emb = model.forward_one(img).detach()

            distances = torch.cdist(q_emb, torch.stack(prototypes))
            pred = classes[torch.argmin(distances)]

            if pred == true_label:
                correct += 1
            total += 1

    accuracy = correct / total
    print(f"{n_way}-way {k_shot}-shot Accuracy: {accuracy:.2%}")
    return accuracy


train(model, train_loader, optimizer, criterion, epochs=10)

evaluate_few_shot(model, test_siamese, n_way=5, k_shot=2)
