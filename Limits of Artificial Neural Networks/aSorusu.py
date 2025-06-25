import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, models
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score  

#rasgelelik için sabitler
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

#verileri yükleme
data = np.load("datasets_labeled/problem_A.npz")
X_train_full = data["X_train"]
y_train_full = data["y_train"]
X_test = data["X_test"]
y_test = data["y_test"]

#CNN için kanal boyutu ekle
X_train_full = X_train_full[..., np.newaxis]
X_test = X_test[..., np.newaxis]

#Veri tipi dönüşümü + normalizasyon
X_train_full = X_train_full.astype("float32") / np.max(X_train_full)
X_test = X_test.astype("float32") / np.max(X_train_full)

#cnn modeli oluşturma fonksiyonu
def create_cnn_model(input_shape=(25,25,1)):
    model = models.Sequential([
        layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(pool_size=2),
        layers.Conv2D(64, kernel_size=3, activation='relu'),
        layers.MaxPooling2D(pool_size=2),
        layers.Conv2D(128, kernel_size=3, activation='relu'),
        layers.MaxPooling2D(pool_size=2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

#eğitim ve değerlendirme fonksiyonu
#param early_stop ile erken durdurma ekleniyor
#param history ile modelin eğitim geçmişi tutulup döndürülüyor
# param r2 ile R² skoru hesaplanıyor
#param preds ile tahmin edilen değerler tutulup döndürülüyor
#param title ile her deneme için başlık veriliyor
#param mae ile test setindeki ortalama mutlak hata hesaplanıyor
def train_and_evaluate(X_train, y_train, X_test, y_test, title):
    model = create_cnn_model(input_shape=X_train.shape[1:])
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        verbose=0,
        validation_data=(X_test, y_test),
        callbacks=[early_stop]
    )
    
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    preds = model.predict(X_test).flatten()
    r2 = r2_score(y_test, preds)

    print(f"{title} - Test MAE: {mae:.4f} | R² Skoru: {r2:.4f}")
    
    return mae, preds, model, r2, history

#sonuçları saklamak için bir sözlük
results = {}
#farklı oranlarda eğitim seti kullanarak denemeler yapılıyor
for ratio in [0.25, 0.5, 1.0]:
    size = int(len(X_train_full) * ratio)#boyut oranına göre eğitim seti boyutu
    X_subset = X_train_full[:size]#size oranına göre eğitim seti alt kümesi
    y_subset = y_train_full[:size]#size oranına göre etiketler alt kümesi
    title = f"%{int(ratio*100)} Veri ({size} örnek)"
    
    mae, preds, model, r2, history = train_and_evaluate(X_subset, y_subset, X_test, y_test, title)
    results[title] = {
        "mae": mae,
        "preds": preds,
        "model": model,
        "r2": r2,
        "history": history
    }

# Tahmin analizi fonksiyonu
def show_predictions(y_true, y_pred, num=5):
    errors = np.abs(y_true - y_pred)# hata hesaplama
    idx_sorted = np.argsort(errors)# hata değerlerine göre sıralama
    
    print("\n✅ En Doğru Tahminler:")
    for i in idx_sorted[:num]:
        print(f"Gerçek: {y_true[i]:.2f} | Tahmin: {y_pred[i]:.2f} | Hata: {errors[i]:.2f}")
    
    print("\n❌ En Yanlış Tahminler:")
    for i in idx_sorted[-num:]:
        print(f"Gerçek: {y_true[i]:.2f} | Tahmin: {y_pred[i]:.2f} | Hata: {errors[i]:.2f}")

for title, res in results.items():
    print(f"\n{title} için tahmin analizi:")
    show_predictions(y_test, res["preds"])

#grafikler

#MAE karşılaştırması
plt.figure()
plt.bar(results.keys(), [r["mae"] for r in results.values()], color='skyblue')
plt.title("Test MAE Karşılaştırması")
plt.ylabel("MAE (Ortalama Mutlak Hata)")
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig("mae_karsilastirma.png")
plt.show()

# tahmin vs gerçek Scatter Plot 
colors = ['blue', 'orange', 'green']
plt.figure(figsize=(18, 5))

for i, (title, res) in enumerate(results.items()):
    plt.subplot(1, len(results), i+1)
    plt.scatter(y_test, res["preds"], alpha=0.6, color=colors[i])
    plt.plot([0, np.max(y_test)], [0, np.max(y_test)], 'r--')
    plt.xlabel("Gerçek Mesafe")
    plt.ylabel("Tahmin Edilen Mesafe")
    plt.title(f"Tahmin vs Gerçek\n{title}")
    plt.grid(True)

plt.tight_layout()
plt.show()

# tüm tahminler birlikte scatter plot
plt.figure(figsize=(6,6))
for (title, res), color in zip(results.items(), colors):
    plt.scatter(y_test, res["preds"], alpha=0.4, label=title, color=color)

plt.plot([0, np.max(y_test)], [0, np.max(y_test)], 'r--')
plt.xlabel("Gerçek Mesafe")
plt.ylabel("Tahmin Edilen Mesafe")
plt.title("Tahmin vs Gerçek (Tüm Veri Setleri)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#R² Skor Karşılaştırması
plt.figure()
plt.bar(results.keys(), [r["r2"] for r in results.values()], color='lightgreen')
plt.title("Test R² Skor Karşılaştırması")
plt.ylabel("R² Skoru")
plt.ylim(0, 1)
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

#loss grafiği
plt.figure(figsize=(10,6))
for title, res in results.items():
    plt.plot(res["history"].history["loss"], label=f"{title} - Train Loss")
    plt.plot(res["history"].history["val_loss"], linestyle='--', label=f"{title} - Val Loss")

plt.title("Eğitim ve Doğrulama Loss Değerleri (Epoch Bazlı)")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_grafik.png")
plt.show()
