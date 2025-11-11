import matplotlib.pyplot as plt
import pandas as pd

pl = pd.read_csv("~/Downloads/lit_pl.csv")
pl.plot(x='wavelength', y='normalization', style='o', label = "Literature data")
plt.show()
