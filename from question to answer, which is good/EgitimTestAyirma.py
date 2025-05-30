import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_excel("ogrenci_sorular_2025.xlsx")
df.dropna(inplace=True)

# %80 eğitim, %20 test olarak böl
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["Hangisi iyi? (1: gpt4o daha iyi, 2: deepseek daha iyi, 3: ikisi de yeterince iyi, 4: ikisi de kötü)"])

#excel olarak kaydetme
train_df.to_excel("train.xlsx", index=False)
test_df.to_excel("test.xlsx", index=False)

print("Veri başarıyla ikiye bölündü ve kaydedildi.")
