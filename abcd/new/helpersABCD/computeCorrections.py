import numpy as np
def computeCorrections(m_fit, m_err, q_fit, q_err, pearson_data_values, cov_matrix_fit):
    corrections = m_fit*pearson_data_values + q_fit
    corrections = 1/corrections
    # Compute uncertainty
    # corrections = 1/(mx+q)
    # dc/dm = -1/(mx+q)^2 * x
    # dc/dq = -1/(mx+q)^2
    der_m = -pearson_data_values/(m_fit*pearson_data_values+q_fit)**2
    der_q = -1/(m_fit*pearson_data_values+q_fit)**2
    err_corrections = np.sqrt(der_m**2*m_err**2 + der_q**2*q_err**2 + 2* der_m*der_q*cov_matrix_fit[1,0])

    return corrections, err_corrections