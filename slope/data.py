import fire
import math
import random
import json

MAX_M = 100
MAX_B = 100

LARGEST_VALUE = 1000

def generate_all_functions():
    data = []
    for m in range(1, MAX_M):
        for b in range(1, MAX_B):
            data.append({ "m": m, "b": b })
    random.shuffle(data)
    return data
def generate_data(functions):
    data = []
    for f in functions:
        b = f["b"]
        m = f["m"]

        max_x = (LARGEST_VALUE  - b) // m
        for x0 in range(1, max_x):
            x1 = x0
            while x1 == x0:
                x1 = random.randint(0, max_x)

            y0 = m * x0 + b
            y1 = m * x1 + b

            data.append({
                "p0": (x0, y0),
                "p1": (x1, y1),
                "m": m,
                "b": b
            })
    random.shuffle(data)
    return data
def main():
    coeffs = generate_all_functions()
    train_f = coeffs[:math.floor(len(coeffs) * 0.8)]
    test_f = coeffs[math.floor(len(coeffs) * 0.8):]

    train_data = generate_data(train_f)
    test_data = generate_data(test_f)

    with open("train.json", "w") as f:
        json.dump(train_data, f)
    with open("test.json", "w") as f:
        json.dump(test_data, f)
    
if __name__ == '__main__':
    fire.Fire(main)
