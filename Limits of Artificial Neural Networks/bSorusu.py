import numpy as np
import tensorflow as tf
from keras import layers, models
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score

# Sabitler
np.random.seed(42)
tf.random.set_seed(42)

#veri yükleme
def load_data(problem_id='B'):
    data = np.load(f"datasets_labeled/problem_{problem_id}.npz")
    X_train_full = data["X_train"]
    y_train_full = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    return X_train_full, y_train_full, X_test, y_test

#veriyi CNN için hazırlama
def prepare_data(X, y):
    X = X[..., np.newaxis].astype(np.float32)
    y = y.astype(np.float32)
    return X, y

# CNN Modeli oluşturma
def build_cnn_model(input_shape=(25,25,1)):
    model = models.Sequential([
        layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(pool_size=2),
        layers.Conv2D(64, kernel_size=3, activation='relu'),
        layers.MaxPooling2D(pool_size=2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

#eğitim ve değerlendirme fonksiyonu
# param early_stop ile erken durdurma ekleniyor
# param history ile modelin eğitim geçmişi tutulup döndürülüyor
# param r2 ile R² skoru hesaplanıyor
# param preds ile tahmin edilen değerler tutulup döndürülüyor
# param title ile her deneme için başlık veriliyor
def train_and_evaluate(X_train_full, y_train_full, X_test, y_test, train_fraction=1.0, title="", epochs=100, batch_size=64):
    n_train = int(len(X_train_full)*train_fraction)
    X_train = X_train_full[:n_train]
    y_train = y_train_full[:n_train]
    
    X_train, y_train = prepare_data(X_train, y_train)
    X_test_prep, y_test_prep = prepare_data(X_test, y_test)
    
    model = build_cnn_model()

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test_prep, y_test_prep),
        callbacks=[early_stop],
        verbose=0
    )
    
    test_loss, test_mae = model.evaluate(X_test_prep, y_test_prep, verbose=0)
    preds = model.predict(X_test_prep).flatten()
    r2 = r2_score(y_test, preds)
    
    print(f"{title} - Test MAE: {test_mae:.4f} | R² Skoru: {r2:.4f}")
    
    return test_mae, preds, model, r2, history

#tahmin analizi
def show_predictions(y_true, y_pred, num=5):
    errors = np.abs(y_true - y_pred)
    idx_sorted = np.argsort(errors)
    
    print("\n✅ En Doğru Tahminler:")
    for i in idx_sorted[:num]:
        print(f"Gerçek: {y_true[i]:.2f} | Tahmin: {y_pred[i]:.2f} | Hata: {errors[i]:.2f}")
    
    print("\n❌ En Yanlış Tahminler:")
    for i in idx_sorted[-num:]:
        print(f"Gerçek: {y_true[i]:.2f} | Tahmin: {y_pred[i]:.2f} | Hata: {errors[i]:.2f}")

#eğitim süreci için grafik çizimi
def plot_training_history(results):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    for title, res in results.items():
        history = res["history"]
        plt.plot(history.history['loss'], label=f'{title} - Train Loss')
        plt.plot(history.history['val_loss'], label=f'{title} - Val Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()

    plt.subplot(1,2,2)
    for title, res in results.items():
        history = res["history"]
        plt.plot(history.history['mae'], label=f'{title} - Train MAE')
        plt.plot(history.history['val_mae'], label=f'{title} - Val MAE')
    plt.title('MAE over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()

    plt.tight_layout()
    plt.show()

#test MAE sonuçlarının grafiği
def plot_test_mae(results):
    plt.figure()
    plt.bar(results.keys(), [r["mae"] for r in results.values()], color='skyblue')
    plt.title("Test MAE Karşılaştırması")
    plt.ylabel("MAE (Ortalama Mutlak Hata)")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

#test R² sonuçlarının grafiği
def plot_test_r2(results):
    plt.figure()
    plt.bar(results.keys(), [r["r2"] for r in results.values()], color='lightgreen')
    plt.title("Test R² Skor Karşılaştırması")
    plt.ylabel("R² Skoru")
    plt.ylim(0, 1)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

#tahmin vs gerçek dağılımı scatter plotları
def plot_scatter_predictions(y_test, results):
    colors = ['blue', 'orange', 'green']
    plt.figure(figsize=(18, 5))
    
    #ayrı ayrı scatter plotlar
    for i, (title, res) in enumerate(results.items()):
        plt.subplot(1, len(results), i+1)
        plt.scatter(y_test, res["preds"], alpha=0.6, color=colors[i])
        plt.plot([0, np.max(y_test)], [0, np.max(y_test)], 'r--')
        plt.xlabel("Gerçek Değer")
        plt.ylabel("Tahmin Edilen Değer")
        plt.title(f"Tahmin vs Gerçek\n{title}")
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    #tüm oranlar bir arada scatter plot
    plt.figure(figsize=(6,6))
    for (title, res), color in zip(results.items(), colors):
        plt.scatter(y_test, res["preds"], alpha=0.4, label=title, color=color)
    plt.plot([0, np.max(y_test)], [0, np.max(y_test)], 'r--')
    plt.xlabel("Gerçek Değer")
    plt.ylabel("Tahmin Edilen Değer")
    plt.title("Tahmin vs Gerçek (Tüm Veri Oranları)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    problem_id = 'B'

    print(f"{problem_id} problemi için veri yükleniyor...")
    X_train_full, y_train_full, X_test, y_test = load_data(problem_id)

    #sonuçları saklamak için bir sözlük
    results = {}

    for ratio in [0.25, 0.5, 1.0]:
        size = int(len(X_train_full) * ratio)#boyutu hesapla
        title = f"%{int(ratio*100)} Veri ({size} örnek)"#başlık oluştur
        
        mae, preds, model, r2, history = train_and_evaluate(
            X_train_full, y_train_full, X_test, y_test, 
            train_fraction=ratio, title=title
        )
        
        results[title] = {
            "mae": mae,
            "preds": preds,
            "model": model,
            "r2": r2,
            "history": history
        }

    #tahmin analizi
    for title, res in results.items():
        print(f"\n{title} için tahmin analizi:")
        show_predictions(y_test, res["preds"])

    #grafikler
    plot_training_history(results)
    plot_test_mae(results)
    plot_test_r2(results)
    plot_scatter_predictions(y_test, results)