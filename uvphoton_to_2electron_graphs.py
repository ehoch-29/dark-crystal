import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ast
import matplotlib.cm as cm


data = pd.read_csv('monitoring_DB.tsv', sep='\t')
data = data.sort_values("ANSAMP")


data["f0"] = data["f0"].apply(ast.literal_eval)
#data["ANSAMP"] = data["ANSAMP"].apply(ast.literal_eval)
#data["EXP"] = data["EXP"].apply(ast.literal_eval)


exposures  = [0, 60]
colors = ['red', 'blue']
for j, n in enumerate(exposures):
    data_exp = data[data["EXP"]==n]
    hdu0 = []
    hdu1 = []
    hdu2 = []
    hdu3 = []
    for i, row in data_exp.iterrows():
        hdu0.append(row["f0"][0])
        hdu1.append(row["f0"][1])
        hdu2.append(row["f0"][2])
        hdu3.append(row["f0"][3])
    
    plt.plot(data_exp["ANSAMP"], hdu0, 'x', color = colors[j], label = 'HDU 0 ' + str(n))
    #plt.plot(data["ANSAMP"], hdu1, 'o', label = 'HDU 1')
    plt.plot(data_exp["ANSAMP"], hdu2, 's', color = colors[j], label = 'HDU 2 ' + str(n))
    plt.plot(data_exp["ANSAMP"], hdu3, '^', color = colors[j], label = 'HDU 3 ' + str(n))

    x = np.arange(0, 420, 20)

    slope, intercept = np.polyfit(data_exp["ANSAMP"], hdu0, 1)
    y = slope*x + intercept
    plt.plot(x, y, ':', color = colors[j], label = 'fit for HDU0')
    print(slope, intercept)
    
    slope, intercept = np.polyfit(data_exp["ANSAMP"], hdu2, 1)
    y = slope*x + intercept
    plt.plot(x, y, ':', color = colors[j], label = 'fit for HDU2')
    print(slope, intercept)
    
    slope, intercept = np.polyfit(data_exp["ANSAMP"], hdu3, 1)
    y = slope*x + intercept
    plt.plot(x, y, ':', color = colors[j], label = 'fit for HDU3')
    print(slope, intercept)


plt.title("Fraction of 0 e- pixels per number of samples")
plt.xlabel("Number of Samples")
plt.ylabel("Fraction of pixels with 0 e'")
plt.legend()
plt.show()
