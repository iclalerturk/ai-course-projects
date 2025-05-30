import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

#bu fonksiyon, iki gömü arasındaki benzerlikleri hesaplar ve en yüksek K benzerlik skoruna sahip indeksleri döndürür
#s_embeddings: kaynak gömü, target_embeddings: hedef gömü, k: en yüksek K benzerlik skoru
#top1: en yüksek benzerlik skoru, top5: en yüksek 5 benzerlik skoru
#n: kaynak gömü uzunluğu
#similarities: kaynak gömü ile tüm hedef gömüler arasındaki kosinüs benzerlikleri
#top_k_indices: en yüksek K benzerlik skoruna sahip indeksler
def calculate_topk(s_embeddings, target_embeddings, k=5):
    top1, top5 = 0, 0
    n = len(s_embeddings)

    for i in tqdm(range(n), desc="Top-K hesaplanıyor"):
        # Kaynak gömü ile tüm hedef gömüler arasındaki kosinüs benzerliklerini hesapla
        similarities = cosine_similarity([s_embeddings[i]], target_embeddings)[0]
        # En yüksek K benzerlik skoruna sahip indeksleri sırala
        top_k_indices = similarities.argsort()[-k:][::-1]
        # Eğer doğru cevap Top-K içinde ise Top-5'e ekle
        if i in top_k_indices:
            top5 += 1
            # Eğer doğru cevap en yüksek skora sahipse Top-1'e ekle
            if i == top_k_indices[0]:
                top1 += 1

    # Top-1 ve Top-5 doğruluk oranlarını döndür
    return top1 / n, top5 / n

# Bu fonksiyon, modelin değerlendirilmesi için kullanılır
# prefix: modelin adı (e5, cosmos, jina)
# base_path: modelin dosya yolu
# s: kaynak gömü, g: gpt4o gömüsü, d: deepseek gömüsü
# gpt4o_top1, gpt4o_top5: gpt4o gömüsü için en yüksek 1 ve 5 benzerlik skoru
# deepseek_top1, deepseek_top5: deepseek gömüsü için en yüksek 1 ve 5 benzerlik skoru
def evaluate_model(prefix):
    base_path = f"embeddings1000/{prefix}_bin_rastgele_soru"
    print(f"\n--- {prefix.upper()} modeli için değerlendirme başlıyor ---")

    s = np.load(f"{base_path}_s.npy")
    g = np.load(f"{base_path}_g.npy")
    d = np.load(f"{base_path}_d.npy")

    gpt4o_top1, gpt4o_top5 = calculate_topk(s, g, k=5)
    print(f"{prefix.upper()} - GPT4O - Top1: {gpt4o_top1*100:.2f}%, Top5: {gpt4o_top5*100:.2f}%")

    deepseek_top1, deepseek_top5 = calculate_topk(s, d, k=5)
    print(f"{prefix.upper()} - DeepSeek - Top1: {deepseek_top1*100:.2f}%, Top5: {deepseek_top5*100:.2f}%")
    
    return {
        'model': prefix.upper(),
        'gpt4o_top1': gpt4o_top1 * 100,
        'gpt4o_top5': gpt4o_top5 * 100,
        'deepseek_top1': deepseek_top1 * 100,
        'deepseek_top5': deepseek_top5 * 100
    }

def visualize_results(results):
    # Create a DataFrame from the results
    df = pd.DataFrame(results)
    
    # Set up the figure and subplots
    plt.figure(figsize=(15, 10))
    
    # Bar chart for Top-1 accuracy
    plt.subplot(2, 1, 1)
    x = np.arange(len(df))
    width = 0.35
    plt.bar(x - width/2, df['gpt4o_top1'], width, label='GPT4O')
    plt.bar(x + width/2, df['deepseek_top1'], width, label='DeepSeek')
    plt.xlabel('Model')
    plt.ylabel('Top-1 Accuracy (%)')
    plt.title('Top-1 Accuracy Comparison')
    plt.xticks(x, df['model'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Bar chart for Top-5 accuracy
    plt.subplot(2, 1, 2)
    plt.bar(x - width/2, df['gpt4o_top5'], width, label='GPT4O')
    plt.bar(x + width/2, df['deepseek_top5'], width, label='DeepSeek')
    plt.xlabel('Model')
    plt.ylabel('Top-5 Accuracy (%)')
    plt.title('Top-5 Accuracy Comparison')
    plt.xticks(x, df['model'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Create and display a table
    table_data = df[['model', 'gpt4o_top1', 'gpt4o_top5', 'deepseek_top1', 'deepseek_top5']]
    table_data.columns = ['Model', 'GPT4O Top-1', 'GPT4O Top-5', 'DeepSeek Top-1', 'DeepSeek Top-5']
    print("\nResults Table:")
    print(table_data.to_string(index=False))

# Tüm modelleri sırayla değerlendir ve sonuçları topla
results = []
for model_prefix in ["e5", "cosmos", "jina"]:
    result = evaluate_model(model_prefix)
    results.append(result)

# Sonuçları görselleştir
visualize_results(results)
