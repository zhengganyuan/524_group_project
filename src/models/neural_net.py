"""
TextCNN Neural Network Model
==============================
Multi-kernel CNN for text classification with GloVe embedding support.
Architecture: Embedding -> Multi-size Conv1d -> Global Max Pool -> FC -> Output
"""

from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class TextCNN(nn.Module):
    """
    TextCNN with multiple kernel sizes to capture n-gram patterns.

    Architecture:
        - 3 kernel sizes [3, 4, 5] each with 128 filters
        - Global max pooling per feature map
        - Dropout + single FC layer for classification
    """

    def __init__(self, vocab_size, embed_dim, num_classes=7,
                 num_filters=128, filter_sizes=(3, 4, 5), dropout=0.5,
                 pretrained_embeddings=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = True

        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, fs)
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        embedded = self.embedding(x).permute(0, 2, 1)  # (batch, embed_dim, seq_len)
        conv_outs = [F.relu(conv(embedded)).max(dim=2).values for conv in self.convs]
        out = self.dropout(torch.cat(conv_outs, dim=1))
        return self.fc(out)


class MentalHealthDataset(Dataset):
    """PyTorch Dataset for tokenized mental health text."""

    def __init__(self, texts, labels, vocab, max_len=256):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.texts[idx].lower().split()
        indices = [self.vocab.get(t, 1) for t in tokens]  # 1 = <UNK>
        indices = indices[:self.max_len]
        indices += [0] * (self.max_len - len(indices))    # 0 = <PAD>
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]


def build_vocab(texts, max_size=20000):
    """Build vocabulary from training texts with frequency-based truncation."""
    counter = Counter()
    for text in texts:
        counter.update(text.lower().split())
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in counter.most_common(max_size):
        vocab[word] = len(vocab)
    print(f"[Vocab] Built vocabulary: {len(vocab)} tokens (from {len(counter)} unique words)")
    return vocab


def load_glove_embeddings(glove_path, vocab, embed_dim=100):
    """
    Load GloVe vectors and build an embedding matrix aligned with vocab.

    Args:
        glove_path: Path to glove.6B.100d.txt
        vocab: dict mapping word -> index
        embed_dim: Embedding dimension (must match GloVe file)

    Returns:
        torch.FloatTensor of shape (vocab_size, embed_dim)
    """
    embeddings = np.random.normal(0, 0.1, (len(vocab), embed_dim))
    embeddings[0] = 0  # <PAD> zero vector

    found = 0
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            if word in vocab:
                embeddings[vocab[word]] = np.array(parts[1:], dtype=np.float32)
                found += 1

    print(f"[GloVe] Loaded {found}/{len(vocab)} word vectors ({found/len(vocab)*100:.1f}% coverage)")
    return torch.tensor(embeddings, dtype=torch.float32)


def create_data_loaders(train_texts, train_labels, test_texts, test_labels,
                        vocab, max_len=256, batch_size=64):
    """Create PyTorch DataLoaders for training and testing."""
    train_dataset = MentalHealthDataset(train_texts, train_labels, vocab, max_len)
    test_dataset = MentalHealthDataset(test_texts, test_labels, vocab, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch. Returns (average_loss, accuracy)."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += inputs.size(0)

    return total_loss / total, correct / total


def evaluate_model(model, data_loader, criterion, device):
    """Evaluate model. Returns (average_loss, accuracy, all_preds, all_labels)."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += inputs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def get_prediction_probs(model, data_loader, device):
    """Get softmax prediction probabilities for ROC-AUC computation."""
    model.eval()
    all_probs = []

    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())

    return np.vstack(all_probs)
