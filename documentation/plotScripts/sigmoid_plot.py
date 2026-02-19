# %%
import matplotlib.pyplot as plt
import numpy as np
# %%
def sigmoid(Z):
    return 1/(1+(np.exp((-Z))))
threshold = 0.5
x = np.linspace(0, 1, 100)
alpha = [2,5,7,10, 20]
fig, ax = plt.subplots()
for a in alpha:
    
    y = sigmoid(a * (x - threshold))
    ax.plot(x, y, label=f'alpha={a}')
ax.legend()
# %%
