import json
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from skopt.space import Real
from skopt.utils import use_named_args
from skopt import gp_minimize

# --- Step 1: Load previous log lines ---
log_path = "/t3home/gcelotto/ggHbb/PNN/NN_highPt/bayes_opt/model_b2/logs_bayes.json"  # Change to your actual log file path

X = []
y = []

with open(log_path, "r") as f:
    for line in f:
        entry = json.loads(line)

        params = entry["params"]
        target = entry["target"]
        x = [
            params["learning_rate"],
            params["batch_size_log2"],
            params["dropout"],
            params["n1_log2"],
            params["n2_log2"],
            params["n3_log2"],
        ]
        X.append(x)
        y.append(-1*target)

X = np.array(X)
y = np.array(y)
print(len(y), " entries")
# --- Step 2: Fit GP regressor ---
kernel = C(1.0, (1e-3, 1e2)) * Matern(nu=2.5) + WhiteKernel()
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True, random_state=42)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
gp.fit(X_scaled, y)

# --- Step 3: Use skopt to find best parameters ---
space = [
    Real(1e-5, 5e-3, name="learning_rate"),
    Real(8, 16, name="batch_size_log2"),
    Real(0.0, 0.5, name="dropout"),
    Real(5, 12, name="n1_log2"),
    Real(4, 10, name="n2_log2"),
    Real(0, 7, name="n3_log2"),
]

@use_named_args(space)
def predict_loss(**params):
    x = np.array([[params[k.name] for k in space]])
    x_scaled = scaler.transform(x)
    pred = gp.predict(x_scaled, return_std=False)
    return pred[0]

# --- Step 4: Minimize predicted loss from GP model ---
# (We use gp_minimize only for convenience, not real function evals)
result = gp_minimize(
    func=predict_loss,
    dimensions=space,
    n_calls=30,
    random_state=42,
    n_initial_points=5,
    acq_func="EI"
)

# --- Step 5: Output best prediction ---
print("\nBest predicted target:", result.fun)
print("Best parameters:")
for dim, val in zip(space, result.x):
    print(f"  {dim.name}: {val:.6f}")
