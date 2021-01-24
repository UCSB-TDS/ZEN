
"""
This is quantized version of cosine similarity: 
https://en.wikipedia.org/wiki/Cosine_similarity
v1, v2: feature vector with 512 elements, each element is a INT8? 
"""
def is_same_identity(v1, v2, threshold=50):
    # cosine = numerator / sart(norm_1 * norm_2)
    norm_1 = sum(i**2 for i in v1)
    norm_2 = sum(i**2 for i in v2)
    numerator = sum(i * j for i, j in zip(v1, v2))

    if numerator < 0:
        return False

    return 10000 * numerator**2 > threshold**2 * norm_1 * norm_2
