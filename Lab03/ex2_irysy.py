import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

BASE_DIR = Path(__file__).resolve().parent

#WCZYTYWANIE DANYCH I NORMALIZACJA

def load_and_prepare_data(csv_path, test_size=0.2, batch_size=16):
    print("WCZYTYWANIE DANYCH...")

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku CSV: {csv_path}")
    
    # Wczytywanie CSV
    df = pd.read_csv(csv_path)
    print(f"Załadowano {len(df)} próbek")
    print(f"Kolumny: {df.columns.tolist()}\n")
    
    # Rozdział features i etykiet
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # Kodowanie etykiet: setosa=0, versicolor=1, virginica=2
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    
    print(f"Klasy: {encoder.classes_}")
    print(f"Liczba klas: {len(encoder.classes_)}\n")
    
    # Podział na train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=test_size, random_state=13, stratify=y_encoded
    )
    
    print(f"Train: {len(X_train)} próbek")
    print(f"Validation: {len(X_val)} próbek\n")
    
    # NORMALIZACJA
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_val_norm = scaler.transform(X_val)
    
    print("Normalizacja:")
    print(f"  Train mean: {X_train_norm.mean(axis=0)}")
    print(f"  Train std: {X_train_norm.std(axis=0)}\n")
    
    # Konwersja na PyTorch tensory
    X_train_tensor = torch.FloatTensor(X_train_norm)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val_norm)
    y_val_tensor = torch.LongTensor(y_val)
    
    # Stworzenie Datasetów
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    # Stworzenie DataLoaderów (batch processing)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, len(encoder.classes_), X_train_norm.shape[1]

#MODEL SIECI NEURONOWEJ

class IrisNeuralNetwork(nn.Module):  
    def __init__(self, input_dim=4, num_classes=3):
        super(IrisNeuralNetwork, self).__init__()
        
        # Warstwy
        self.fc1 = nn.Linear(input_dim, 16)      # 4 -> 16
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(16, 8)               # 16 -> 8
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(8, num_classes)      # 8 -> 3
    
    def forward(self, x):
        # Propagacja do przodu
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x

# PĘTLA TRENINGOWA

def train_epoch(model, train_loader, criterion, optimizer, device):
    #Trening dla jednej epoki
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metryki
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    #Walidacja modelu
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


def train_model(model, train_loader, val_loader, epochs=100, lr=0.01, device='cpu'):
    # Główna pętla treningowa
    print("\nTRENING MODELU...")
    print(f"Epochs: {epochs}, Learning Rate: {lr}\n")
    
    criterion = nn.CrossEntropyLoss()  # Zawiera log_softmax + NLLLoss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Historia dla wizualizacji
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    for epoch in range(1, epochs + 1):
        # Trening
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Walidacja
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Output co 10 epok
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    
    return train_losses, train_accs, val_losses, val_accs



# WIZUALIZACJA PROCESÓW UCZENIA

def plot_learning_curves(train_losses, val_losses, train_accs, val_accs):
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Wykres 1: Loss
    axes[0].plot(train_losses, label='Train Loss', linewidth=2, color='#2E86AB')
    axes[0].plot(val_losses, label='Validation Loss', linewidth=2, color='#A23B72')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (CrossEntropy)', fontsize=12)
    axes[0].set_title('Loss Curves: Train vs Validation', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Wykres 2: Accuracy
    axes[1].plot(train_accs, label='Train Accuracy', linewidth=2, color='#2E86AB')
    axes[1].plot(val_accs, label='Validation Accuracy', linewidth=2, color='#A23B72')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Accuracy Curves: Train vs Validation', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    output_path = BASE_DIR / 'learning_curves.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Wykresy uczenia zapisane: {output_path}")
    if matplotlib.get_backend().lower() != 'agg':
        plt.show()

# STATYSTYKI WALIDACJI + MACIERZ BŁĘDU

def print_validation_stats(y_true, y_pred, class_names):

    print("\n" + "="*70)
    print("FINALNE STATYSTYKI WALIDACJI")
    print("="*70)
    
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy (całkowita dokładność): {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Macierz błędu
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nMacierz Błędu (Confusion Matrix):")
    print(f"     {' '.join([f'{class_names[i]:>12}' for i in range(len(class_names))])}")
    for i, row in enumerate(cm):
        print(f"{class_names[i]:>6} {str(row)}")
    
    # Bardziej ładna wizualizacja macierzy
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Liczba próbek'})
    plt.title('Confusion Matrix - Iris Classification', fontsize=14, fontweight='bold')
    plt.ylabel('Prawdziwa klasa', fontsize=12)
    plt.xlabel('Predykowana klasa', fontsize=12)
    output_path = BASE_DIR / 'confusion_matrix.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Macierz błędu zapisana: {output_path}")
    if matplotlib.get_backend().lower() != 'agg':
        plt.show()
    
    # Szczegółowy raport
    print(f"\nSzczegółowy classification report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

# CZĘŚĆ F: INTERPRETACJA WYNIKÓW

def print_interpretation(y_true, y_pred, val_acc, train_acc_final, val_acc_final):

    print("\n" + "="*70)
    print("INTERPRETACJA WYNIKÓW")
    print("="*70)
    
    print(f"""
1. WYDAJNOŚĆ MODELU:
   - Train Accuracy (ostatnia epoka): {train_acc_final:.4f}
   - Val Accuracy (ostatnia epoka):   {val_acc_final:.4f}
   
   Prognoza: Jeśli obie są wysokie (~0.95+) → model dobrze uogólnia!
             Jeśli Train >> Val → overfitting (model pamiętał dane)
             Jeśli oba niskie → underfitting (model zbyt prosty)

2. FUNKCJA STRATY (CrossEntropy):
   - Mierzy rozbieżność między predykcjami a prawdą
   - Maleje → model się uczy
   - Powinniśmy zobaczyć spadek szczególnie na początkowych epokach

3. DANE:
   - Iris to klasyczny dataset do ML
   - 3 klasy: setosa, versicolor, virginica
   - 4 cechy: długość/szerokość kielicha i płatka
   - Klasy są zdefiiniowane mało się przekrywają → klasyfikacja dość prosta!

4. ARCHITEKTURA DECYZJA:
   - Sieć 4-16-8-3 (względnie mała) wystarczy dla tego prostego problemu
   - ReLU zapewnia nieliniowość (bez tego = linear model)
   - Dropout zmniejsza overfitting
   - 3 warstwy to "głębokie" uczenie (deep learning)

5. CROSS-ENTROPY LOSS:
   - Idealny dla klasyfikacji wieloklasowej
   - Penalizuje złe predykcje bardziej niż dobre
   - Log-softmax + NLLLoss w jednym

6. HIPERPARAMETRY:
   - lr (learning rate) = 0.01: szybkość uczenia
   - batch_size = 16: ile próbek w jednym kroku
   - dropout = 0.2: losowo zerujemy 20% połączeń
   - Walidacja na 20% danych
    """)

# MAIN

if __name__ == "__main__":
    # Ustawienia
    CSV_PATH = Path(__file__).resolve().parent / "iris_big.csv"
    EPOCHS = 150
    BATCH_SIZE = 16
    LEARNING_RATE = 0.01
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Device: {DEVICE}")
    print(f"PyTorch version: {torch.__version__}\n")
    
    # A) Wczytanie danych
    train_loader, val_loader, num_classes, input_dim = load_and_prepare_data(
        CSV_PATH, test_size=0.2, batch_size=BATCH_SIZE
    )
    
    # B) Stworzenie modelu
    print("BUDOWA MODELU...")
    model = IrisNeuralNetwork(input_dim=input_dim, num_classes=num_classes)
    model = model.to(DEVICE)
    print(f"Model:\n{model}\n")
    
    # C) Trening
    train_losses, train_accs, val_losses, val_accs = train_model(
        model, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE, device=DEVICE
    )
    
    # D) Wizualizacja
    print("\nTWORZENIE WYKRESÓW...")
    plot_learning_curves(train_losses, val_losses, train_accs, val_accs)
    
    # E) Finalna walidacja + Macierz błędu
    val_loss, val_acc, y_pred, y_true = validate(
        model, val_loader, nn.CrossEntropyLoss(), DEVICE
    )
    class_names = ['setosa', 'versicolor', 'virginica']
    print_validation_stats(y_true, y_pred, class_names)
    
    # F) Interpretacja
    print_interpretation(y_true, y_pred, val_acc, train_accs[-1], val_accs[-1])
    
    print("\nTRENING ZAKOŃCZONY!")
