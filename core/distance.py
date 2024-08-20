from cfg import np

def distance_l1 (x, y) :
    return abs(x - y)

def distance_l2 (x,y):
    return (x-y)**2

def cosine_similarity (x, y) :
    numerator = np.dot (x, y)
    denominator = np. linalg . norm (x) * np. linalg . norm (y)

    return numerator / denominator
