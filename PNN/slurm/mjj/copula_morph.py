from scipy.stats import norm
import numpy as np

def copula_morph(X, qt_tt, qt_ggH, L_tt, L_ggH, eps=1e-6):

    # CR to uniform
    U = qt_tt.transform(X)
    U = np.clip(U, eps, 1 - eps)

    # uniform to normal
    Z = norm.ppf(U)

    # whiten using fixed CR matrix
    # This operationw whitens  Z under the assumption that it has the same covariance as ttbar
    # if not, it will not be perfectly white
    Z_white = np.linalg.solve(L_tt, Z.T).T

    # recolor with SR covariance
    Z_corr = Z_white @ L_ggH.T

    # normal to uniform
    U_corr = norm.cdf(Z_corr)

    # uniform to SR marginals
    X_morphed = qt_ggH.inverse_transform(U_corr)

    return X_morphed