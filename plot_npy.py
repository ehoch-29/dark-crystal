import numpy as np
import matplotlib.pyplot as plt
import sys

file_path = sys.argv[1]
#file_path_2 = sys.argv[2]
data = np.load(file_path)
#data_2 = np.load(file_path_2)

# Check the shape of the data
print(f"Data shape: {data.shape}")
data_trans = data.T
#data_trans_2 = data_2.T
plt.plot(data_trans[0], data_trans[1]*10e9)
#plt.plot(data_trans_2[0], data_trans_2[1]*10e9, label = "covered")
plt.title(str(file_path))

plt.xlabel("Wavelength (nm)")
plt.ylabel("Power (nanowatts)")
plt.show()
# Plot the data
if data.ndim == 1:
    # If the data is 1D
    plt.plot(data)
    plt.title("1D Data Plot")
    plt.xlabel("Index")
    plt.ylabel("Value")
elif data.ndim == 2:
    # If the data is 2D
    plt.imshow(data, aspect='auto', cmap='viridis')
    plt.title("2D Data Heatmap")
    plt.colorbar(label="Value")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
else:
    print("Data has more than 2 dimensions. Cannot plot directly.")
    exit()

plt.show()
