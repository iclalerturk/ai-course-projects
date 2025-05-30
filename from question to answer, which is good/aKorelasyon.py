import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Korelasyon analizi fonksiyonu
# s_embeddings: kaynak gömü, target_embeddings: hedef gömü
# labels: etiketler, model_name: model adı, target_name: hedef adı
# k: en yüksek K benzerlik skoru
#stats: istatistikleri tutacak sözlük
# example_results: örnek bazlı sonuçları tutacak liste  
#highest_similarity_indices: en yüksek benzerlikteki cevapları bulma
# correct_top_k: Top-K doğruluğu, correct_top_1: Top-1 doğruluğu
# example_top_1_accuracies: örnek bazlı Top-1 doğruluk oranları
# Bu fonksiyon, iki gömü arasındaki benzerlikleri hesaplar ve en yüksek K benzerlik skoruna sahip indeksleri döndürür
# Ayrıca, doğruluk oranlarını ve korelasyonları hesaplar ve görselleştirir
def analyze_correlation(s_embeddings, target_embeddings, labels, model_name, target_name, k=5):

    # İstatistikleri tutacak sözlük
    stats = {i: {'total': 0, 'correct_top1': 0, 'correct_topk': 0} for i in range(1, 5)}
    
    # Benzerlik hesaplama
    similarities = cosine_similarity(s_embeddings, target_embeddings)
    
    # En yüksek benzerlikteki cevapları bulma
    top_k_indices = np.argsort(similarities, axis=1)[:, -k:]
    highest_similarity_indices = np.argmax(similarities, axis=1)
    
    # Doğruluk kontrolü
    correct_top_k = []
    correct_top_1 = []
    example_top_1_accuracies = []
    example_top_k_accuracies = []
    
    # Örnek bazlı sonuçları tutacak liste
    example_results = []
    
    for i in range(len(s_embeddings)):
        # Top-1 doğruluğu
        is_correct_top_1 = (highest_similarity_indices[i] == i)
        correct_top_1.append(is_correct_top_1)
        example_top_1_accuracies.append(1.0 if is_correct_top_1 else 0.0)
        
        # Top-k doğruluğu
        is_correct_top_k = (i in top_k_indices[i])
        correct_top_k.append(is_correct_top_k)
        example_top_k_accuracies.append(1.0 if is_correct_top_k else 0.0)
        
        # Örnek bazlı sonuçları kaydet
        example_results.append({
            'example_index': i,
            'class': labels[i],
            'top1_correct': is_correct_top_1,
            'topk_correct': is_correct_top_k,
            'similarity_score': similarities[i, i],
            'top_k_indices': top_k_indices[i].tolist()
        })
    
    # İstatistikleri güncelle
    for i in range(len(s_embeddings)):
        class_idx = labels[i]
        stats[class_idx]['total'] += 1
        if correct_top_1[i]:
            stats[class_idx]['correct_top1'] += 1
        if correct_top_k[i]:
            stats[class_idx]['correct_topk'] += 1
    
    # Korelasyon hesaplama
    pearson_top1 = np.corrcoef(similarities.diagonal(), correct_top_1)[0, 1]
    pearson_topk = np.corrcoef(similarities.diagonal(), correct_top_k)[0, 1]
    spearman_top1 = spearmanr(similarities.diagonal(), correct_top_1)[0]
    spearman_topk = spearmanr(similarities.diagonal(), correct_top_k)[0]
    
    # Örnek bazlı ve toplu doğruluk hesaplama
    overall_top1_accuracy = sum(correct_top_1) / len(correct_top_1)
    overall_topk_accuracy = sum(correct_top_k) / len(correct_top_k)
    
    # Sonuçları yazdır
    print(f"\n{model_name} - {target_name} Korelasyon Sonuçları:")
    print(f"Top-1 Başarı Oranı: {overall_top1_accuracy:.2%}")
    print(f"Top-{k} Başarı Oranı: {overall_topk_accuracy:.2%}")
    print(f"Top-1 Pearson Korelasyonu: {pearson_top1:.3f}")
    print(f"Top-{k} Pearson Korelasyonu: {pearson_topk:.3f}")
    print(f"Top-1 Spearman Korelasyonu: {spearman_top1:.3f}")
    print(f"Top-{k} Spearman Korelasyonu: {spearman_topk:.3f}")
    
    # Örnek bazlı sonuçları yazdır
    print("\nÖrnek Bazlı Sonuçlar (İlk 10 örnek):")
    print("Örnek | Sınıf | Top-1 | Top-5 | Benzerlik Skoru")
    print("-" * 50)
    for result in example_results[:10]:
        print(f"{result['example_index']:6d} | {result['class']:5d} | {result['top1_correct']:5} | {result['topk_correct']:5} | {result['similarity_score']:.4f}")
    
    # Sınıf bazlı başarı oranlarını hesapla
    class_success_rates = []
    for class_idx in range(1, 5):
        total = stats[class_idx]['total']
        if total > 0:
            top1_rate = stats[class_idx]['correct_top1'] / total
            topk_rate = stats[class_idx]['correct_topk'] / total
            class_success_rates.append((class_idx, top1_rate, topk_rate))
    
    # Sınıf bazlı korelasyonları hesapla
    class_correlations = []
    for class_idx in range(1, 5):
        class_indices = [i for i, label in enumerate(labels) if label == class_idx]
        if len(class_indices) > 0:
            class_similarities = similarities.diagonal()[class_indices]
            class_correct_top1 = [correct_top_1[i] for i in class_indices]
            class_correct_topk = [correct_top_k[i] for i in class_indices]
            
            pearson_top1_cls = np.corrcoef(class_similarities, class_correct_top1)[0, 1]
            pearson_topk_cls = np.corrcoef(class_similarities, class_correct_topk)[0, 1]
            spearman_top1_cls = spearmanr(class_similarities, class_correct_top1)[0]
            spearman_topk_cls = spearmanr(class_similarities, class_correct_topk)[0]
            
            class_correlations.append((class_idx, pearson_top1_cls, pearson_topk_cls, 
                                     spearman_top1_cls, spearman_topk_cls))
    
    # Sınıf bazlı sonuçları yazdır
    print("\nSınıf Bazlı Başarı Oranları:")
    for class_idx, top1_rate, topk_rate in class_success_rates:
        print(f"Sınıf {class_idx}:")
        print(f"  Top-1 Başarı Oranı: {top1_rate:.2%}")
        print(f"  Top-{k} Başarı Oranı: {topk_rate:.2%}")
    
    print("\nSınıf Bazlı Korelasyonlar:")
    for class_idx, pearson1, pearsonk, spearman1, spearmank in class_correlations:
        print(f"Sınıf {class_idx}:")
        print(f"  Top-1 Pearson Korelasyonu: {pearson1:.3f}")
        print(f"  Top-{k} Pearson Korelasyonu: {pearsonk:.3f}")
        print(f"  Top-1 Spearman Korelasyonu: {spearman1:.3f}")
        print(f"  Top-{k} Spearman Korelasyonu: {spearmank:.3f}")
    
    # Görselleştirme
    plt.figure(figsize=(20, 15))
    
    # 1. Sınıf bazlı başarı oranları
    plt.subplot(3, 2, 1)
    classes = [str(c[0]) for c in class_success_rates]
    top1_rates = [c[1] for c in class_success_rates]
    topk_rates = [c[2] for c in class_success_rates]
    
    x = np.arange(len(classes))
    width = 0.35
    
    plt.bar(x - width/2, top1_rates, width, label=f'Top-1', color='skyblue')
    plt.bar(x + width/2, topk_rates, width, label=f'Top-{k}', color='lightgreen')
    
    plt.xlabel('Sınıf')
    plt.ylabel('Başarı Oranı')
    plt.title(f'{model_name} - {target_name} Sınıf Bazlı Başarı Oranları')
    plt.xticks(x, classes)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Örnek bazlı başarı oranları
    plt.subplot(3, 2, 2)
    plt.plot(example_top_1_accuracies, label='Top-1', color='skyblue', alpha=0.7)
    plt.plot(example_top_k_accuracies, label=f'Top-{k}', color='lightgreen', alpha=0.7)
    plt.xlabel('Örnek İndeksi')
    plt.ylabel('Doğruluk')
    plt.title(f'{model_name} - {target_name} Örnek Bazlı Doğruluk')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Benzerlik dağılımı
    plt.subplot(3, 2, 3)
    plt.hist(similarities.diagonal(), bins=30, alpha=0.7, color='purple')
    plt.xlabel('Benzerlik Skoru')
    plt.ylabel('Frekans')
    plt.title(f'{model_name} - {target_name} Benzerlik Dağılımı')
    plt.grid(True, alpha=0.3)
    
    # 4. Korelasyon görselleştirmesi
    plt.subplot(3, 2, 4)
    plt.scatter(similarities.diagonal(), correct_top_1, alpha=0.5, label='Top-1', color='skyblue')
    plt.scatter(similarities.diagonal(), correct_top_k, alpha=0.5, label=f'Top-{k}', color='lightgreen')
    plt.xlabel('Benzerlik Skoru')
    plt.ylabel('Doğruluk')
    plt.title(f'{model_name} - {target_name} Korelasyon')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Örnek bazlı benzerlik skorları
    plt.subplot(3, 2, 5)
    example_indices = np.arange(len(example_results))
    similarity_scores = [r['similarity_score'] for r in example_results]
    top1_correct = [r['top1_correct'] for r in example_results]
    
    plt.scatter(example_indices, similarity_scores, c=top1_correct, cmap='coolwarm', alpha=0.7)
    plt.colorbar(label='Top-1 Doğruluk')
    plt.xlabel('Örnek İndeksi')
    plt.ylabel('Benzerlik Skoru')
    plt.title(f'{model_name} - {target_name} Örnek Bazlı Benzerlik Skorları')
    plt.grid(True, alpha=0.3)
    
    # 6. Sınıf bazlı benzerlik dağılımları
    plt.subplot(3, 2, 6)
    for class_idx in range(1, 5):
        class_scores = [r['similarity_score'] for r in example_results if r['class'] == class_idx]
        if class_scores:
            sns.kdeplot(class_scores, label=f'Sınıf {class_idx}', fill=True, alpha=0.3)
    
    plt.xlabel('Benzerlik Skoru')
    plt.ylabel('Yoğunluk')
    plt.title(f'{model_name} - {target_name} Sınıf Bazlı Benzerlik Dağılımları')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'overall_top1_accuracy': overall_top1_accuracy,
        'overall_topk_accuracy': overall_topk_accuracy,
        'example_top1_accuracies': example_top_1_accuracies,
        'example_topk_accuracies': example_top_k_accuracies,
        'pearson_top1': pearson_top1,
        'pearson_topk': pearson_topk,
        'spearman_top1': spearman_top1,
        'spearman_topk': spearman_topk,
        'class_success_rates': class_success_rates,
        'class_correlations': class_correlations,
        'example_results': example_results
    }

# etiketleri okuma
df = pd.read_excel("bin_rastgele_soru.xlsx")
labels = df['Hangisi iyi? (1: gpt4o daha iyi, 2: deepseek daha iyi, 3: ikisi de yeterince iyi, 4: ikisi de kötü)'].to_numpy()

#Tüm modelleri sırayla analiz etme
for model_prefix in ["e5", "cosmos", "jina"]:
    base_path = f"embeddings1000/{model_prefix}_bin_rastgele_soru"
    print(f"\n--- {model_prefix.upper()} modeli için analiz başlıyor ---")

    s = np.load(f"{base_path}_s.npy")
    g = np.load(f"{base_path}_g.npy")
    d = np.load(f"{base_path}_d.npy")

    analyze_correlation(s, g, labels, model_name=model_prefix, target_name="GPT4O")
    analyze_correlation(s, d, labels, model_name=model_prefix, target_name="DeepSeek")
