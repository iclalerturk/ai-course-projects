from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import torch

#bu fonksiyon, farklı embedding modelleri kullanarak metin verilerinin gömülerini çıkarır ve kaydeder
# model_name: kullanılacak embedding modeli
# prefix: dosya adı için ön ek
#file: embedding çıkarılacak dosya adı (örneğin, bin_rastgele_soru)
# df: embedding çıkarılacak veri çerçevesi
def embed_model(model_name, prefix):
    print(f"\n--- Model yükleniyor: {model_name} ---")
    
    # gpu kontrolü
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"GPU kullanılıyor: {torch.cuda.get_device_name(0)}")
    
    # modeli cihazla yükleme
    model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
    
    #embedding çıkarma
    for file in ["bin_rastgele_soru"]:
        df = pd.read_excel(f"{file}.xlsx")
        print(f"{prefix}_{file} için embedding çıkarılıyor...")
        # Veriyi encode et ve GPU kullanımı aktif tut
        s = model.encode(df["Soru"].tolist(), normalize_embeddings=True, device=device)
        np.save(f"embeddings1000/{prefix}_{file}_s.npy", s)
        del s #bellekte yer kaplamaması için silme

        print("soru embedding çıkarıldı!")
        g = model.encode(df["gpt4o cevabı"].tolist(), normalize_embeddings=True, device=device)
        np.save(f"embeddings1000/{prefix}_{file}_g.npy", g)
        del g
        print("gpt4o cevabı embedding çıkarıldı!")
        d = model.encode(df["deepseek cevabı"].tolist(), normalize_embeddings=True, device=device)
        print("deepseek cevabı embedding çıkarıldı!")
        
        np.save(f"embeddings1000/{prefix}_{file}_d.npy", d)
        del d
        print(f"{prefix}_{file} için embedding çıkarıldı!")
        print(f"{file} için embedding çıkarıldı ve kaydedildi: {model_name}")

# Embedding modellerini sırayla uygula
embed_model("intfloat/multilingual-e5-large-instruct", "e5")
embed_model("ytu-ce-cosmos/turkish-e5-large", "cosmos")
embed_model("jinaai/jina-embeddings-v3", "jina")
