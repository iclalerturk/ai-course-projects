import numpy as np
import tensorflow as tf
from keras import layers, models, callbacks
import matplotlib.pyplot as plt
from itertools import product

# Sabitler
np.random.seed(42)
tf.random.set_seed(42)

#veri yükleme
def load_data(problem_id='E'):
    data = np.load(f"datasets_labeled/problem_{problem_id}.npz")
    return data["X_train"], data["y_train"], data["X_test"], data["y_test"]
#veriyi MLP için hazırlama
def prepare_data_mlp(X, y):
    X = X.reshape((X.shape[0], -1)).astype(np.float32)
    y = y.astype(np.float32)
    return X, y

# MLP Modeli oluşturma
def build_mlp_model(input_dim=625, hidden_units=[128, 64], dropout_rate=0.3):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(input_dim,)))
    for units in hidden_units:
        model.add(layers.Dense(units, activation='relu'))
        model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Eğitim ve değerlendirme fonksiyonu
# param early_stop ile erken durdurma ekleniyor
# param history ile modelin eğitim geçmişi tutulup döndürülüyor
# param test_mae ile test setindeki ortalama mutlak hata hesaplanıyor
# param preds ile tahmin edilen değerler tutulup döndürülüyor
def train_model(X_train, y_train, X_test, y_test, hidden_units, dropout_rate, batch_size, epochs=30):
    X_train_prep, y_train_prep = prepare_data_mlp(X_train, y_train)
    X_test_prep, y_test_prep = prepare_data_mlp(X_test, y_test)

    model = build_mlp_model(hidden_units=hidden_units, dropout_rate=dropout_rate)
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(X_train_prep, y_train_prep,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_test_prep, y_test_prep),
                        callbacks=[early_stop],
                        verbose=0)

    test_loss, test_mae = model.evaluate(X_test_prep, y_test_prep, verbose=0)
    return model, history, test_mae

#uygun hiperparametre arama fonksiyonu
#param unit_options, dropout_options, batch_options ile denenecek hiperparametreler belirleniyor
#param product ile tüm kombinasyonlar oluşturuluyor
#param best_mae ile en düşük MAE tutuluyor
#param best_config ile en iyi hiperparametreler tutuluyor
##param best_model ile en iyi model tutuluyor
#param test_mae ile test setindeki ortalama mutlak hata hesaplanıyor
def hyperparameter_search(X_train, y_train, X_test, y_test):
    unit_options = [[128, 64], [64, 32]]
    dropout_options = [0.2, 0.3]
    batch_options = [16, 32]

    best_mae = float('inf')
    best_config = None
    best_model = None

    for units, dropout, batch in product(unit_options, dropout_options, batch_options):
        print(f"Deneme: units={units}, dropout={dropout}, batch_size={batch}")
        model, _, test_mae = train_model(X_train, y_train, X_test, y_test,
                                         hidden_units=units,
                                         dropout_rate=dropout,
                                         batch_size=batch)
        print(f"MAE: {test_mae:.4f}")

        if test_mae < best_mae:
            best_mae = test_mae
            best_config = {'hidden_units': units, 'dropout_rate': dropout, 'batch_size': batch}
            best_model = model

    print(f"\n En iyi hiperparametreler: {best_config} | En düşük MAE: {best_mae:.4f}")
    return best_model, best_config

#tahminlerin analizi fonksiyonu
# param n_samples ile en iyi ve en kötü tahmin sayısı belirleniyor
# param prepare_data_mlp ile veriler hazırlanıyor
# param model.predict ile tahminler alınıyor
# param np.argsort ile en iyi ve en kötü tahminlerin indeksleri bulunuyor
def analyze_predictions_mlp(model, X_test, y_test, n_samples=5):
    X_test_prep, y_test_prep = prepare_data_mlp(X_test, y_test)
    preds = model.predict(X_test_prep).flatten()
    errors = np.abs(preds - y_test)

    best_idxs = np.argsort(errors)[:n_samples]
    worst_idxs = np.argsort(errors)[-n_samples:]

    print("\n En iyi tahminler:")
    for i in best_idxs:
        print(f"Gerçek: {y_test[i]:.1f} | Tahmin: {preds[i]:.2f} | Hata: {errors[i]:.2f}")

    print("\n En kötü tahminler:")
    for i in worst_idxs:
        print(f"Gerçek: {y_test[i]:.1f} | Tahmin: {preds[i]:.2f} | Hata: {errors[i]:.2f}")

    plt.figure(figsize=(6,6))
    plt.scatter(y_test, preds, alpha=0.5)
    plt.plot([0, np.max(y_test)], [0, np.max(y_test)], 'r--')
    plt.xlabel("Gerçek Kare Sayısı")
    plt.ylabel("Tahmin")
    plt.title("Tahmin vs Gerçek")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#grafik çizim fonksiyonları
def plot_training_history(histories, fractions):
    plt.figure(figsize=(12,5))
    for frac, history in zip(fractions, histories):
        plt.plot(history.history['val_mae'], label=f'Val MAE (%{int(frac*100)})')
    plt.title("Validation MAE (Farklı Eğitim Oranları)")
    plt.xlabel("Epoch")
    plt.ylabel("Val MAE")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def plot_test_mae(results):
    plt.bar([f"%{int(k*100)}" for k in results.keys()], list(results.values()), color='skyblue')
    plt.ylabel("Test MAE")
    plt.title("Eğitim Oranına Göre Test MAE")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

def compare_scatter_plots_by_fraction(models, fractions, X_test, y_test):
    X_test_prep, _ = prepare_data_mlp(X_test, y_test)
    y_true = y_test

    plt.figure(figsize=(18, 5))
    
    for i, (model, frac) in enumerate(zip(models, fractions), 1):
        preds = model.predict(X_test_prep).flatten()
        
        plt.subplot(1, len(fractions), i)
        plt.scatter(y_true, preds, alpha=0.5)
        plt.plot([0, np.max(y_true)], [0, np.max(y_true)], 'r--')
        plt.xlabel("Gerçek")
        plt.ylabel("Tahmin")
        plt.title(f"Veri %{int(frac*100)} ile Eğitim")
        plt.grid(True)

    plt.tight_layout()
    plt.show()

def calculate_accuracy_mlp(model, X_test, y_test):
    X_test_prep, y_test_prep = prepare_data_mlp(X_test, y_test)
    preds = model.predict(X_test_prep).flatten()
    preds_rounded = np.round(preds).astype(int)
    y_true = y_test.astype(int)
    accuracy = np.mean(preds_rounded == y_true)
    print(f"🎯 Accuracy (tam sayıya yuvarlanmış): {accuracy*100:.2f}%")
    return accuracy


if __name__ == "__main__":
    print("E problemi için veri yükleniyor...")
    X_train_full, y_train_full, X_test, y_test = load_data('E')

    print("\n🔍 Hiperparametre optimizasyonu başlatılıyor...")
    _, best_params = hyperparameter_search(X_train_full, y_train_full, X_test, y_test)

    train_fractions = [0.25, 0.5, 1.0]# farklı eğitim oranları
    histories = []# eğitim geçmişlerini saklamak için
    results = {}# sonuçları saklamak için

    models_by_frac = []# farklı eğitim oranları için modelleri saklamak için

    for frac in train_fractions:
        print(f"\n Eğitim oranı %{int(frac*100)} ile model eğitiliyor...")
        N = int(len(X_train_full) * frac)
        model, history, mae = train_model(
            X_train_full[:N], y_train_full[:N],
            X_test, y_test,
            hidden_units=best_params['hidden_units'],
            dropout_rate=best_params['dropout_rate'],
            batch_size=best_params['batch_size']
        )
        histories.append(history)
        results[frac] = mae
        models_by_frac.append(model)


    plot_training_history(histories, train_fractions)
    plot_test_mae(results)

    print("\n En iyi modelle detaylı analiz:")
    model_full, _, _ = train_model(
        X_train_full, y_train_full, X_test, y_test,
        hidden_units=best_params['hidden_units'],
        dropout_rate=best_params['dropout_rate'],
        batch_size=best_params['batch_size']
    )

    analyze_predictions_mlp(model_full, X_test, y_test)
    calculate_accuracy_mlp(model_full, X_test, y_test)
    compare_scatter_plots_by_fraction(models_by_frac, train_fractions, X_test, y_test)

