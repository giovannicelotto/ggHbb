# %%
import matplotlib.pyplot as plt
import numpy as np
import mplhep as hep
hep.style.use("CMS")
def scatter_hist(x, y, ax, ax_histx, ax_histy):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y)

    # now determine nice limits by hand:
    binwidth = 0.25
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins, edgecolor='black')
    ax_histy.hist(y, bins=bins, orientation='horizontal', edgecolor='black')
# %%
x = np.random.normal(loc=0, scale=1, size=1000)
y = np.random.normal(loc=0, scale=1, size=1000)
fig = plt.figure(layout='constrained')
# Create the main Axes.
ax = fig.add_subplot()
# The main Axes' aspect can be fixed.
ax.set_aspect('equal')
ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
ax_histy = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)
# Draw the scatter plot and marginals.
scatter_hist(x, y, ax, ax_histx, ax_histy)
ax.set_xlabel("Feature x")
ax.set_ylabel("Feature y")
plt.show()
# %%

y = 0.4*x + np.sqrt(1-0.4**2)*y
fig = plt.figure(layout='constrained')
ax = fig.add_subplot()
ax.set_aspect('equal')

ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
ax_histy = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)
# Draw the scatter plot and marginals.
scatter_hist(x, y, ax, ax_histx, ax_histy)
ax.set_xlabel("Feature x")
ax.set_ylabel("Feature y")
plt.show()
# %%
import numpy as np
import matplotlib.pyplot as plt

# Assume x and y are defined
# Stack them as columns
data = np.vstack([x, y])

# 1. Compute covariance matrix
cov = np.cov(data)
print("Covariance matrix:\n", cov)

# 2. Cholesky decomposition
L = np.linalg.cholesky(cov)
print("Cholesky factor L:\n", L)

# 3. Whiten the data
# Whitened data = L^-1 @ data
data_whitened = np.linalg.inv(L) @ data

# Split back into x and y
x_whitened, y_whitened = data_whitened

# 4. Plot the whitened data
fig, ax = plt.subplots(1, 1)
ax.set_aspect('equal')

ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
ax_histy = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)


scatter_hist(x_whitened, y_whitened, ax, ax_histx, ax_histy)

ax.set_xlabel("Whitened Feature x")
ax.set_ylabel("Whitened Feature y")

plt.show()
# %%
# Apply it twice for the fun of it
data_whitened = np.linalg.inv(L) @ data_whitened
x_whitened, y_whitened = data_whitened
fig, ax = plt.subplots(1, 1)
ax.set_aspect('equal')
ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
ax_histy = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)
scatter_hist(x_whitened, y_whitened, ax, ax_histx, ax_histy)
ax.set_xlabel("Recolored Feature x")
ax.set_ylabel("Recolored Feature y")

plt.show()
# %%
