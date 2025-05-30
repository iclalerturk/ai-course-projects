import numpy as np
import pygad
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

# Parametreler
POPULATION_SIZES = np.arange(10, 110, 20)  # popülasyon boyutları 10 ile 100 arasında 20'şer artışla
MUTATION_RATES = np.linspace(0.1, 0.9, 5)  # mutasyon oranları 0.1 ile 0.9 arasında 9 eşit aralıklı değer
GEN_COUNT = 100  # jenerasyon sayısı

def get_patterns(): # Bu fonksiyon 7 adet 3x3'lük pattern döndürür
    return np.array([
        [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        [[1, 0, 1], [0, 0, 0], [1, 0, 1]],
        [[0, 0, 0], [1, 1, 1], [0, 0, 0]],
        [[0, 1, 0], [0, 1, 0], [0, 1, 0]],
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    ], dtype=np.int8)

def apply_patterns(image, patterns): # bu fonksiyon istenen resme en uygun patternleri uygular
    h, w = image.shape
    reconstructed = np.zeros((h, w)) # yeniden oluşturulmuş görüntü için boş bir matris 
    # 3x3'lük blokları tarayarak en uygun patterni bul ve uygula
    for i in range(0, h, 3):
        for j in range(0, w, 3):
            block = image[i:i+3, j:j+3] #resmin 3x3'lük blokları
            best_pattern = min(patterns, key=lambda p: np.sum(np.abs(p - block))) # en uygun patterni bul
            reconstructed[i:i+3, j:j+3] = best_pattern # en uygun patterni yeniden oluşturulmuş görüntüye uygula
    return reconstructed

def load_images_from_png(file_paths):# bu fonksiyon png dosyalarından resimleri yükler
    images = []
    for path in file_paths:
        img = Image.open(path).convert("L")
        img = img.resize((24, 24))
        img = np.array(img) // 255
        images.append(img)
    return np.array(images)

def visualize_patterns(patterns): # bu fonksiyon patternleri görselleştirir
    fig, axes = plt.subplots(1, 7, figsize=(10, 3))
    for i, ax in enumerate(axes):
        ax.imshow(patterns[i], cmap='gray')
        ax.axis('off')
    plt.show()

def fitness_func(ga_instance, solution, solution_idx): # bu fonksiyon uygunluk fonksiyonudur
    if ga_instance.generations_completed == 0: #ilk jenerasyonda oluşturulan patternler kullanılır
        patterns = get_patterns() 
    else:
        patterns = [np.array(solution[i*9:(i+1)*9]).reshape(3,3) for i in range(7)] # çözümü 3x3'lük patternlere dönüştür
    
    total_loss = 0
    # her görüntü için uygun pattern uygulanır ve loss hesaplanır
    for img in images:
        reconstructed = apply_patterns(img, patterns)
        total_loss += np.sum(np.abs(img - reconstructed))
    return -total_loss# toplam hata negatif olarak döndürülür çünkü pygad en büyük değeri arar

def run_experiment(pop_size, mutation_rate):# bu fonksiyon genetik algoritmayı çalıştırır ve sonuçları döndürür
    ga_instance = pygad.GA( 
        num_generations=GEN_COUNT, # toplam jenerasyon sayısı
        num_parents_mating=int(pop_size/2), # ebeveynlerin sayısı
        fitness_func=fitness_func, # uygunluk fonksiyonu
        sol_per_pop=int(pop_size), # # popülasyondaki çözüm sayısı
        # her çözüm 7 adet 3x3'lük patternden oluşur, her bir pattern 9 elemanlıdır
        num_genes=7*9,
        gene_type=int, # gen türü
        gene_space=[0, 1],# gen aralığı 0 ve 1
        mutation_probability=mutation_rate, # mutasyon oranı
        save_best_solutions=True, # en iyi çözümleri kaydet
        suppress_warnings=True #gürültyü azaltmak için uyarıları bastırır
    )
    ga_instance.run() # genetik algoritmayı çalıştırır

    best_solution = ga_instance.best_solution()[0] # en iyi çözümü alır
    # en iyi çözümü 3x3'lük patternlere dönüştür
    best_patterns = [np.array(best_solution[i*9 : (i+1)*9]).reshape(3, 3) for i in range(7)]
    # her görüntü için en iyi pattern uygulanır ve toplam hata hesaplanır
    total_loss = sum(np.sum(np.abs(img - apply_patterns(img, best_patterns))) for img in images)
    fitness_history = ga_instance.best_solutions_fitness # en iyi çözümlerin uygunluk değerleri için bir liste döndürür

    return fitness_history, total_loss, best_patterns  # fitness geçmişi, toplam hata ve en iyi patternleri döndürür

def general_analysis(): # genel analiz fonksiyonu 50 popülasyon ve 0.1 mutasyon oranı ve 100 jenerasyon ile çalıştırılır
    # Sabit parametrelerle deneyi çalıştır
    pop_size = 50
    mutation_rate = 0.1
    ga_instance = pygad.GA(
        num_generations=GEN_COUNT,# toplam jenerasyon sayısı
        num_parents_mating=int(pop_size/2), # ebeveynlerin sayısı
        fitness_func=fitness_func, # uygunluk fonksiyonu
        sol_per_pop=int(pop_size), # popülasyondaki çözüm sayısı
        num_genes=7*9, # her çözüm 7 adet 3x3'lük patternden oluşur, her bir pattern 9 elemanlıdır
        gene_type=int, # gen türü
        gene_space=[0, 1], # gen aralığı 0 ve 1
        mutation_probability=mutation_rate, # mutasyon oranı
        save_best_solutions=True, # en iyi çözümleri kaydet
        suppress_warnings=True #gürültüyü azaltmak için uyarıları bastırır
    )
    ga_instance.run()

    # Fitness değişim grafiği
    plt.figure(figsize=(10, 5))
    plt.plot(range(GEN_COUNT+1), ga_instance.best_solutions_fitness)
    plt.title("Jenerasyonlara Göre Fitness Değişimi")
    plt.xlabel("Jenerasyon")
    plt.ylabel("Fitness")
    plt.grid()
    plt.show()

    # Pattern evrim analizi
    def track_pattern_evolution(ga_instance):#Jenerasyonlardaki pattern değişimini izler ve görselleştirir
        # her 25 jenerasyonda bir patternleri seç
        selected_generations = np.linspace(0, GEN_COUNT-1, 5, dtype=int)
        pattern_snapshots = []
        
        for gen in selected_generations:
            solution = ga_instance.best_solutions[gen]
            patterns = [solution[i*9:(i+1)*9].reshape(3,3) for i in range(7)]
            pattern_snapshots.append(patterns)
        
        # pattern evrimini görselleştir
        fig, axes = plt.subplots(len(selected_generations), 7, figsize=(20, 15))
        for i, gen in enumerate(selected_generations):
            for j in range(7):
                axes[i,j].imshow(pattern_snapshots[i][j], cmap='gray')
                axes[i,j].set_title(f"Jenerasyon {gen}\nPattern {j+1}")
                axes[i,j].axis('off')
        plt.suptitle("Pattern Evrimi (Zamana Göre)", y=1.02, fontsize=16)
        plt.tight_layout()
        plt.show()
        
        # değişim istatistiklerini hesaplar
        changes = []
        for gen in range(1, GEN_COUNT):
            prev = ga_instance.best_solutions[gen-1]
            current = ga_instance.best_solutions[gen]
            changes.append(np.mean(np.abs(current - prev)))
        # değişim oranını görselleştir
        plt.figure(figsize=(10, 4))
        plt.plot(range(1, GEN_COUNT), changes)
        plt.title("Jenerasyonlar Arası Pattern Değişim Oranı")
        plt.xlabel("Jenerasyon")
        plt.ylabel("Ortalama Pixel Değişim Miktarı")
        plt.grid()
        plt.show()
    
    # pattern evrim analizini çalıştırır
    track_pattern_evolution(ga_instance)

    # en iyi patternleri ve sonuçları gösterir
    best_solution = ga_instance.best_solution()[0]
    best_patterns = [np.array(best_solution[i*9:(i+1)*9]).reshape(3,3) for i in range(7)]
    
    # her görüntü için en iyi pattern uygulanır ve toplam hata hesaplanır
    total_loss = sum(np.sum(np.abs(img - apply_patterns(img, best_patterns))) for img in images)
    print(f"\nToplam Hata: {total_loss}")
    #en iyi patternleri görselleştirir
    visualize_patterns(best_patterns)
    print("\nEn iyi patternler:")
    for i, p in enumerate(best_patterns, 1):
        print(f"\nPattern {i}:")
        print(p)

    # orijinal ve yeniden oluşturulmuş görüntüleri karşılaştır
    last_images = []
    for idx, img in enumerate(images):
        reconstructed = apply_patterns(img, best_patterns)
        loss = np.sum(np.abs(img - reconstructed))
        print(f"\nGörüntü {idx+1} Hata: {loss}")
        last_images.append(reconstructed)
    
    fig, axes = plt.subplots(2, len(images), figsize=(18, 6))
    for i in range(len(images)):
        axes[0,i].imshow(images[i], cmap='gray')
        axes[0,i].set_title(f"Orijinal {i+1}")
        axes[0,i].axis('off')
        
        axes[1,i].imshow(last_images[i], cmap='gray')
        axes[1,i].set_title(f"Yeniden Oluşturulmuş\n(Hata: {np.sum(np.abs(images[i]-last_images[i]))})")
        axes[1,i].axis('off')
    plt.tight_layout()
    plt.show()

def analyze_jenerasyon(): # bu fonksiyon 50 popülasyon ve 0.1 mutasyon oranı ile çalıştırılır
    plt.figure(figsize=(10, 5))
    
    # Sabit parametrelerle deneyi çalıştır
    fitness_history, total_loss, best_patterns = run_experiment(50, 0.1)
    
    plt.plot(range(GEN_COUNT+1), fitness_history)
    plt.title("Jenerasyonlara Göre Hata Değişimi")
    plt.xlabel("Jenerasyon")
    plt.ylabel("Hata")
    plt.grid()
    plt.show()

    # resimleri yeniden oluştur ve hata hesapla
    last_images = []
    for idx, img in enumerate(images):
        reconstructed = apply_patterns(img, best_patterns)
        last_images.append(reconstructed)

    visualize_patterns(best_patterns)
    print("En iyi patternler:")
    for p in best_patterns:
        print(p)

    # orijinal ve yeniden oluşturulmuş görüntüleri karşılaştır
    fig, axes = plt.subplots(2, len(last_images), figsize=(24, 24))
    for i in range(len(last_images)):
        axes[0, i].imshow(last_images[i], cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(images[i], cmap='gray')
        axes[1, i].axis('off')
    plt.show()

def analyze_mutation_effect(): #genetik algoritmadaki mutasyon oranının etkisini analiz edilir 50 popülasyon ve 100 jenerasyon ile
    results = [] # mutasyon oranı ve toplam hata için sonuçları sakla
    best_patterns_list = []  # en iyi patternleri her mutasyon oranı için sakla
    
    for mutation_rate in MUTATION_RATES:
        # run_experiment fonksiyonunu çalıştır ve en iyi patternler ile toplam hatayı al
        _, total_loss, best_patterns = run_experiment(50, mutation_rate)
        results.append((mutation_rate, total_loss))
        best_patterns_list.append(best_patterns)  # en iyi patternleri sakla
    
    # Sonuçları DataFrame'e dönüştür ve çiz
    mutation_df = pd.DataFrame(results, columns=["mutation_rate", "total_loss"])
    plt.figure(figsize=(6, 4))
    plt.plot(mutation_df["mutation_rate"], mutation_df["total_loss"], marker="o")
    plt.title("Mutasyon Oranı vs. Toplam Hata")
    plt.xlabel("Mutasyon Oranı")
    plt.ylabel("Toplam Hata")
    plt.grid()
    plt.show()

    #en küçük toplam hata için en iyi patternleri al
    best_idx = np.argmin([x[1] for x in results])
    best_patterns = best_patterns_list[best_idx]    
    visualize_patterns(best_patterns) # en iyi patternleri görselleştir
    print("En iyi patternler:")
    for p in best_patterns:
        print(p)

    # yeni oluşturulmuş görüntüleri ve hataları hesapla
    last_images = []
    for idx, img in enumerate(images):
        reconstructed = apply_patterns(img, best_patterns)
        last_images.append(reconstructed)

    # Görselleştirme
    fig, axes = plt.subplots(2, len(last_images), figsize=(24, 24))
    for i in range(len(last_images)):
        axes[0, i].imshow(last_images[i], cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(images[i], cmap='gray')
        axes[1, i].axis('off')
    plt.show()

def analyze_population_effect(): #popülasyon büyüklüğünün etkisini analiz eder mutasyon oranı 0.1 ve 100 jenerasyon ile
    results = [] # popülasyon büyüklüğü ve toplam hata için sonuçları sakla
    best_patterns_list = []  # en iyi patternleri her popülasyon büyüklüğü için sakla 
    
    for pop_size in POPULATION_SIZES: # her popülasyon büyüklüğü için run_experiment fonksiyonunu çalıştır ve sonuçları al
        _, total_loss, best_patterns = run_experiment(pop_size, 0.1)
        results.append((pop_size, total_loss))
        best_patterns_list.append(best_patterns)  
    
    # Sonuçları DataFrame'e dönüştür ve çiz
    pop_df = pd.DataFrame(results, columns=["pop_size", "total_loss"])
    plt.figure(figsize=(6, 4))
    plt.plot(pop_df["pop_size"], pop_df["total_loss"], marker="s", color="r")
    plt.title("Popülasyon Büyüklüğü vs. Toplam Hata")
    plt.xlabel("Popülasyon Büyüklüğü")
    plt.ylabel("Toplam Hata")
    plt.grid()
    plt.show()

    # en küçük toplam hata için en iyi patternleri al
    best_idx = np.argmin([x[1] for x in results])
    best_patterns = best_patterns_list[best_idx]
    
    # en iyi patternleri görselleştir
    visualize_patterns(best_patterns) 
    print(f"En iyi patternler (Popülasyon = {POPULATION_SIZES[best_idx]}):")
    for p in best_patterns:
        print(p)

    last_images = []
    for idx, img in enumerate(images):
        reconstructed = apply_patterns(img, best_patterns)
        last_images.append(reconstructed)

    # orijinal ve yeniden oluşturulmuş görüntüleri karşılaştır
    fig, axes = plt.subplots(2, len(last_images), figsize=(24, 24))
    for i in range(len(last_images)):
        axes[0, i].imshow(last_images[i], cmap='gray')
        axes[0, i].set_title(f"Reconstructed (Loss: {np.sum(np.abs(images[i] - last_images[i])):.1f})")
        axes[0, i].axis('off')
        axes[1, i].imshow(images[i], cmap='gray')
        axes[1, i].set_title("Original")
        axes[1, i].axis('off')
    plt.tight_layout()
    plt.show()

# Ana program akışı
print("Genetik Algoritma ile Pattern Bulma Deneyi")
dd=1

while dd==1:
    flag =1
    #deney için resim kümesi seçimi
    while flag == 1:
        print("Deney için resim kümesi seçin:\n")
        print("Küme 1 doğrusal çizgiler \nKüme 2 eğrisel çizgiler\nKüme 3 karmaşık geometrik şekiller\nÇıkış için 0")
        choice0 = int(input())
        if choice0 == 1:
            file_paths = ["set1_image1.png", "set1_image2.png", "set1_image3.png", "set1_image4.png", "set1_image5.png"]
            flag = 0
        elif choice0 == 2:
            file_paths = ["set2_image1.png", "set2_image2.png", "set2_image3.png", "set2_image4.png", "set2_image5.png"]
            flag = 0
        elif choice0 == 3:
            file_paths = ["set3_image1.png", "set3_image2.png", "set3_image3.png", "set3_image4.png", "set3_image5.png"]
            flag = 0
        elif choice0 == 0:
            print("Çıkılıyor...")
            flag = 0
            dd=0
        else:
            print("Geçersiz seçim!")
    if dd!=0:            
        flag = 1
        while flag == 1: # deney için analiz türü seçimi
            images = load_images_from_png(file_paths)
            print("Deney sonuçlarını analiz etmek için aşağıdaki seçeneklerden birini seçin:")
            print("1.Genel olarak uygun pattern bulma\n2. Jenerasyonlara göre fitness değişimi\n3. Mutasyon oranı etkisi\n4. Popülasyon etkisi\nÇıkış için 0")
            print("Seçiminizi yapın (1-4):")
            choice = int(input())
            if choice == 1:
                general_analysis()
                flag = 0
            elif choice == 2:
                analyze_jenerasyon()
                flag = 0
            elif choice == 3:
                analyze_mutation_effect()
                flag = 0
            elif choice == 4:
                analyze_population_effect()
                flag = 0
            elif choice == 0:
                print("Çıkılıyor...")
                flag = 0
                dd=0
            else:
                print("Geçersiz seçim!")

