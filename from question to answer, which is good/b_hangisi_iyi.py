import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore") 

#embeddingleri yükleme fonksiyonu
#prefix: embedding modelinin adı (e5, cosmos, jina)
#s_train, g_train, d_train: soru, gpt40, deepseek eğitim verileri
#s_test, g_test, d_test: soru, gpt40, deepseek test verileri
def load_embeddings(prefix):
    s_train = np.load(f"embeddingsTumveri/{prefix}_train_s.npy")
    g_train = np.load(f"embeddingsTumveri/{prefix}_train_g.npy")
    d_train = np.load(f"embeddingsTumveri/{prefix}_train_d.npy")

    s_test = np.load(f"embeddingsTumveri/{prefix}_test_s.npy")
    g_test = np.load(f"embeddingsTumveri/{prefix}_test_g.npy")
    d_test = np.load(f"embeddingsTumveri/{prefix}_test_d.npy")

    return s_train, g_train, d_train, s_test, g_test, d_test

#istenen kombinasyonları oluşturma
#combination: hangi kombinasyonun oluşturulacağı (s, g, d, s-g, s-d, g-d, |s-g|, |s-g|-|s-d|, concat_sgd)
#s, g, d: embedding vektörleri
#s: soru embeddingi, g: gpt40 embeddingi, d: deepseek embeddingi
def create_combinations(combination, s, g, d):
    if combination == "s":
        return s
    elif combination == "g":
        return g
    elif combination == "d":
        return d
    elif combination == "s-g":
        return s - g
    elif combination == "s-d":
        return s - d
    elif combination == "g-d":
        return g - d
    elif combination == "|s-g|":
        return np.abs(s - g)
    elif combination == "|s-g|-|s-d|":
        return np.abs(s - g) - np.abs(s - d)
    elif combination == "concat_sgd":
        return np.concatenate([s, g, d], axis=1)
    else:
        raise ValueError(f"Geçersiz kombinasyon: {combination}")

# etiketleri yükleme
y_train = pd.read_excel("train.xlsx")[
    "Hangisi iyi? (1: gpt4o daha iyi, 2: deepseek daha iyi, 3: ikisi de yeterince iyi, 4: ikisi de kötü)"].values

y_test = pd.read_excel("test.xlsx")[
    "Hangisi iyi? (1: gpt4o daha iyi, 2: deepseek daha iyi, 3: ikisi de yeterince iyi, 4: ikisi de kötü)"].values

embedding_models = ["e5", "cosmos", "jina"]
combinations = ["s", "g", "d", "s-g", "s-d", "g-d", "|s-g|", "|s-g|-|s-d|", "concat_sgd"]

#tüm sonuçları toplamak için liste
results = []

#eğitim döngüsü
# model_name: embedding modeli (e5, cosmos, jina)
# combo: kombinasyon (s, g, d, s-g, s-d, g-d, |s-g|, |s-g|-|s-d|, concat_sgd)
# X_train: eğitim verileri, y_train: etiketler
# X_test: test verileri, y_test: etiketler
# s_tr, g_tr, d_tr: eğitim verileri (soru, gpt40, deepseek)
# s_ts, g_ts, d_ts: test verileri (soru, gpt40, deepseek)
for model_name in embedding_models:
    print(f"\n Embedding Modeli: {model_name.upper()}")
    s_tr, g_tr, d_tr, s_ts, g_ts, d_ts = load_embeddings(model_name)

    for combo in combinations:

        X_train = create_combinations(combo, s_tr, g_tr, d_tr)
        X_test = create_combinations(combo, s_ts, g_ts, d_ts)

        #regression
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"Kombinasyon: {combo:15} | Doğruluk: {acc*100:.2f}%")

        results.append({
            "Embedding Modeli": model_name,
            "Kombinasyon": combo,
            "Model": "LogisticRegression",
            "Doğruluk (%)": round(acc * 100, 2)
        })

#sonuçları görselleştir
results_df = pd.DataFrame(results)

# 1. Heatmap: Her model ve kombinasyon için doğruluk oranları
plt.figure(figsize=(15, 8))
pivot_df = results_df.pivot(index="Embedding Modeli", columns="Kombinasyon", values="Doğruluk (%)")
sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=.5)
plt.title("Model ve Kombinasyonlara Göre Doğruluk Oranları (%)")
plt.tight_layout()
plt.show()

# 2. Bar plot: Her model için en iyi kombinasyon
plt.figure(figsize=(12, 6))
for model in embedding_models:
    model_results = results_df[results_df["Embedding Modeli"] == model]
    plt.bar(model_results["Kombinasyon"], model_results["Doğruluk (%)"], label=model.upper())
plt.title("Her Model İçin Kombinasyonların Performansı")
plt.xlabel("Kombinasyon")
plt.ylabel("Doğruluk (%)")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# 3. Tablo: Tüm sonuçlar
print("\nTüm Sonuçlar:")
print(results_df.to_string(index=False))

# 4. En iyi sonuçları göster
best_results = results_df.groupby("Embedding Modeli").apply(lambda x: x.nlargest(3, "Doğruluk (%)"))
print("\nHer Model İçin En İyi 3 Kombinasyon:")
print(best_results.to_string(index=False))

# 5. En iyi 3 kombinasyon görselleştirmesi
plt.figure(figsize=(15, 10))

# Her model için ayrı subplot
# Her model için en iyi 3 kombinasyonu al ve görselleştir
#model_best: her model için en iyi 3 kombinasyonu tutar

for i, model in enumerate(embedding_models):
    plt.subplot(3, 1, i+1)
    model_best = best_results.loc[model]
    
    # Bar plot
    bars = plt.bar(model_best["Kombinasyon"], model_best["Doğruluk (%)"], 
                  color='skyblue', width=0.6)
    
    # Her bar üzerine değerleri yaz
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.2f}%', ha='center', va='bottom')
    
    plt.title(f"{model.upper()} Modeli - En İyi 3 Kombinasyon")
    plt.ylabel("Doğruluk (%)")
    plt.ylim(0, max(model_best["Doğruluk (%)"]) + 5)  # Üst sınırı ayarla
    plt.grid(True, alpha=0.3)
    
    # X ekseni etiketlerini döndür
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Tablo formatında da göster
print("\nHer Model İçin En İyi 3 Kombinasyon:")
for model in embedding_models:
    print(f"\n{model.upper()} Modeli:")
    model_best = best_results.loc[model]
    for _, row in model_best.iterrows():
        print(f"  {row['Kombinasyon']:15} : {row['Doğruluk (%)']:.2f}%")
