# DataFlair Iris Flower Classification
# Import Packages
import sqlite3
#import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
import pandas as pd

iris = pd.read_csv("data/Iris.csv",sep=',')
print(iris.head())
# Ścieżka do pliku SQLite w folderze projektu
sqlite_path = 'data/database.sqlite'

# Połączenie z bazą danych SQLite
conn = sqlite3.connect(sqlite_path)



conn.close()
