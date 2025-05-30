import pandas as pd

df = pd.read_excel("ogrenci_sorular_2025.xlsx") 

#boş hücreleri temizleme
df.dropna(inplace=True)

#rastgele 1000 satır seçme
df_sample = df.sample(n=1000, random_state=42)

# yeni bir Excel dosyasına kaydet
df_sample.to_excel("bin_rastgele_soru.xlsx", index=False)
