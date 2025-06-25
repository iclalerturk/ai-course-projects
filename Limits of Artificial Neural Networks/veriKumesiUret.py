import numpy as np
import os

np.random.seed(42)
SAVE_DIR = "datasets_labeled"
os.makedirs(SAVE_DIR, exist_ok=True)

def generate_point_matrix(size=25, num_points=2):
    matrix = np.zeros((size, size), dtype=np.float32)
    coords = []
    for _ in range(num_points):
        while True:
            x, y = np.random.randint(0, size, 2)
            if matrix[x, y] == 0:
                matrix[x, y] = 1
                coords.append((x, y))
                break
    return matrix, coords

def euclidean(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def generate_square_matrix(size=25, num_squares=3, min_size=2, max_size=6):
    matrix = np.zeros((size, size), dtype=np.float32)
    for _ in range(num_squares):
        w, h = np.random.randint(min_size, max_size+1, 2)
        x, y = np.random.randint(0, size - w), np.random.randint(0, size - h)
        matrix[x:x+w, y:y+h] = 1
    return matrix

def generate_sample(problem_type):
    if problem_type == 'A':
        mat, pts = generate_point_matrix(num_points=2)
        label = euclidean(pts[0], pts[1])

    elif problem_type == 'B':
        n = np.random.randint(3, 11)
        mat, pts = generate_point_matrix(num_points=n)
        dists = [euclidean(p1, p2) for i, p1 in enumerate(pts) for p2 in pts[i+1:]]
        label = min(dists)

    elif problem_type == 'C':
        n = np.random.randint(3, 11)
        mat, pts = generate_point_matrix(num_points=n)
        dists = [euclidean(p1, p2) for i, p1 in enumerate(pts) for p2 in pts[i+1:]]
        label = max(dists)

    elif problem_type == 'D':
        n = np.random.randint(1, 11)
        mat, _ = generate_point_matrix(num_points=n)
        label = n

    elif problem_type == 'E':
        n = np.random.randint(1, 11)
        mat = generate_square_matrix(num_squares=n)
        label = n

    else:
        raise ValueError("Geçersiz problem tipi.")
    
    return mat, label

def generate_dataset(problem_type, n_train=800, n_test=200):
    X_train, y_train = [], []
    X_test, y_test = [], []

    for _ in range(n_train):
        x, y = generate_sample(problem_type)
        X_train.append(x)
        y_train.append(y)

    for _ in range(n_test):
        x, y = generate_sample(problem_type)
        X_test.append(x)
        y_test.append(y)

    return (np.array(X_train), np.array(y_train),
            np.array(X_test), np.array(y_test))

# Tüm problemler için üretim ve kaydetme
problem_ids = ['A', 'B', 'C', 'D', 'E']

for pid in problem_ids:
    print(f"Problem {pid} için veri üretiliyor...")
    X_train, y_train, X_test, y_test = generate_dataset(pid)
    
    np.savez_compressed(f"{SAVE_DIR}/problem_{pid}.npz",
                        X_train=X_train, y_train=y_train,
                        X_test=X_test, y_test=y_test)

print("Tüm veri kümeleri başarıyla oluşturuldu ve kaydedildi.")

data = np.load("datasets_labeled/problem_C.npz")
X_train = data["X_train"]
y_train = data["y_train"]
