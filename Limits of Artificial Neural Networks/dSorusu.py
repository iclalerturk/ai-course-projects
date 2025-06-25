import numpy as np
import tensorflow as tf
from keras import layers, models, callbacks
import matplotlib.pyplot as plt

#sabitler
np.random.seed(42)
tf.random.set_seed(42)

#veri yükleme
def load_data(problem_id='D'):
    data = np.load(f"datasets_labeled/problem_{problem_id}.npz")
    X_train_full = data["X_train"]
    y_train_full = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    return X_train_full, y_train_full, X_test, y_test

#veriyi RNN için hazırlama (25 zaman adımı, 25 özellik)
def prepare_data_rnn(X, y):
    X = X.astype(np.float32)
    # RNN input shape: (num_samples, timesteps, features)
    # Matrisimizi 25 satır (timesteps), 25 sütun (features) şeklinde düşün
    X = X.reshape((-1, 25, 25))
    y = y.astype(np.float32)
    return X, y

#RNN Modeli oluşturma (LSTM kullanarak)
def build_rnn_model(input_shape=(25,25), lstm_units=64, dropout_rate=0.2):
    model = models.Sequential([
        layers.LSTM(lstm_units, activation='tanh', input_shape=input_shape),
        layers.Dropout(dropout_rate),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

#eğitim ve değerlendirme fonksiyonu
# param early_stop ile erken durdurma ekleniyor
# param history ile modelin eğitim geçmişi tutulup döndürülüyor
# param test_mae ile test setindeki ortalama mutlak hata hesaplanıyor
# param preds ile tahmin edilen değerler tutulup döndürülüyor
# param train_fraction ile eğitim verisi oranı ayarlanıyor

def train_and_evaluate_rnn(X_train_full, y_train_full, X_test, y_test,
                           train_fraction=1.0, epochs=100, batch_size=16):
    n_train = int(len(X_train_full)*train_fraction)
    indices = np.random.permutation(len(X_train_full))
    X_train = X_train_full[indices[:n_train]]
    y_train = y_train_full[indices[:n_train]]


    X_train, y_train = prepare_data_rnn(X_train, y_train)
    X_test_prep, y_test_prep = prepare_data_rnn(X_test, y_test)

    model = build_rnn_model()
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_test_prep, y_test_prep),
                        callbacks=[early_stop],
                        verbose=2)
    
    test_loss, test_mae = model.evaluate(X_test_prep, y_test_prep, verbose=0)
    print(f"Train fraction: {train_fraction:.2f}, Test MAE: {test_mae:.4f}")

    preds = model.predict(X_test_prep).flatten()

    return model, history, test_mae, preds
# Scatter plot ile tahminlerin analizi
def plot_scatter_all(y_test, all_preds, fractions):
    plt.figure(figsize=(15, 4))
    for i, frac in enumerate(fractions):
        plt.subplot(1, len(fractions), i+1)
        plt.scatter(y_test, all_preds[frac], alpha=0.5, s=10)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel("Gerçek")
        plt.ylabel("Tahmin")
        plt.title(f"%{int(frac*100)} Eğitim Verisi")
        plt.grid(True)
    plt.tight_layout()
    plt.suptitle("Gerçek vs. Tahmin (Scatter Plot)", fontsize=16, y=1.05)
    plt.show()

# Tahmin analizi
def analyze_predictions_rnn(model, X_test, y_test, n_samples=5):
    X_test_prep, y_test_prep = prepare_data_rnn(X_test, y_test)
    preds = model.predict(X_test_prep).flatten()
    errors = np.abs(preds - y_test_prep)

    best_idxs = np.argsort(errors)[:n_samples]
    worst_idxs = np.argsort(errors)[-n_samples:]

    print("\nEn iyi tahminler (Gerçek - Tahmin):")
    for i in best_idxs:
        print(f"{y_test[i]:.1f} - {preds[i]:.2f} (Hata: {errors[i]:.2f})")

    print("\nEn kötü tahminler (Gerçek - Tahmin):")
    for i in worst_idxs:
        print(f"{y_test[i]:.1f} - {preds[i]:.2f} (Hata: {errors[i]:.2f})")

#grafik çizimi
def plot_training_history(histories, fractions):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    for frac, history in zip(fractions, histories):
        plt.plot(history.history['loss'], label=f'Train loss ({int(frac*100)}%)')
        plt.plot(history.history['val_loss'], label=f'Val loss ({int(frac*100)}%)')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()

    plt.subplot(1,2,2)
    for frac, history in zip(fractions, histories):
        plt.plot(history.history['mae'], label=f'Train MAE ({int(frac*100)}%)')
        plt.plot(history.history['val_mae'], label=f'Val MAE ({int(frac*100)}%)')
    plt.title('MAE over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_test_mae(results):
    fractions = list(results.keys())
    maes = list(results.values())

    plt.bar([f"{int(f*100)}%" for f in fractions], maes)
    plt.xlabel("Eğitim Verisi Oranı")
    plt.ylabel("Test MAE")
    plt.title("Farklı Eğitim Veri Oranlarının Test Performansı")
    plt.show()

if __name__ == "__main__":
    problem_id = 'D'

    print(f"{problem_id} problemi için veri yükleniyor...")
    X_train_full, y_train_full, X_test, y_test = load_data(problem_id)


    train_fractions = [0.25, 0.5, 1.0]
    histories = []# Liste olarak tüm eğitim geçmişlerini tutacak
    results = {}# Sözlük olarak her eğitim oranı için test MAE değerlerini tutacak

    all_preds = {}# Sözlük olarak her eğitim oranı için tahminleri tutacak

    for frac in train_fractions:
        print(f"\nEğitim verisi oranı: %{int(frac*100)} ile eğitim başlıyor.")
        model, history, test_mae, preds = train_and_evaluate_rnn(
            X_train_full, y_train_full, X_test, y_test,
            train_fraction=frac)

        histories.append(history)
        results[frac] = test_mae
        all_preds[frac] = preds
 
    #grafikler
    plot_training_history(histories, train_fractions)
    plot_test_mae(results)
    plot_scatter_all(y_test, all_preds, train_fractions)


   



