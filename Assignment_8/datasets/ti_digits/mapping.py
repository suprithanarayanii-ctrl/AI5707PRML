import random

with open("Data/ExtractedData/mapping.txt", "w") as f:
    for i in range(1, 22):
        f.writelines(f"{i} : {random.sample(list(range(1, 10)), 3)}\n")