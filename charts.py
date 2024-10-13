import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import pandas as pd
iris = pd.read_csv("data/Iris.csv",sep=',')
x = iris.drop("Species", axis=1)

print(iris.head()) #This function returns the first n rows for the object based on position.

print(iris.describe()) #Generate descriptive statistics.
print("Target Labels", iris["Species"].unique()) #Return unique values based on a hash table.

z = iris.drop(columns=["Id"])
z.hist(figsize=(10, 8))
plt.show()

fig = px.scatter(iris, x="SepalWidthCm", y="SepalLengthCm", color="Species") #A scatter plot of y vs. x with varying marker size and/or color.
fig.show()

sns.pairplot(iris, hue="Species")
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
sns.boxplot(x="Species", y="SepalLengthCm", data=iris, ax=axes[0, 0])
sns.boxplot(x="Species", y="SepalWidthCm", data=iris, ax=axes[0, 1])
sns.boxplot(x="Species", y="PetalLengthCm", data=iris, ax=axes[1, 0])
sns.boxplot(x="Species", y="PetalWidthCm", data=iris, ax=axes[1, 1])
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(x.drop(columns=["Id"]).corr(), annot=True, cmap="coolwarm")
plt.show()
