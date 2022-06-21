import os

base_path = os.path.split(os.path.realpath(__file__))[0]

print(base_path)

aa = os.path.join(base_path, 'data', 'PPI', 'ppi_1.csv.gz')
print(aa)