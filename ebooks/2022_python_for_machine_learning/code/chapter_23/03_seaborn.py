import matplotlib.pyplot as plt
import seaborn as sns

data = sns.load_dataset("iris")
sns.pairplot(data, kind="scatter", diag_kind="kde", hue="species",
             palette="muted", plot_kws={'alpha':0.7})
plt.show()
