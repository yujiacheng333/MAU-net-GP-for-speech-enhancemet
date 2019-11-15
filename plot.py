import numpy as np
with open("./haha.txt", "r") as f:
    a = f.readlines()
a = a[0].split(",")
print(a)
a = np.asarray(a)[:-1].astype(np.float)

with open("./haha.txt", "r") as f:
    b = f.readlines()
b = b[0].split(",")
print(b)
b = np.asarray(b)[:-1].astype(np.float)

import matplotlib.pyplot as plt
plt.plot(a[-5000:],label="haha")
plt.plot(b[-5000:])
plt.show()