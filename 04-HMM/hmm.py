# Author: Kaituo Xu, Fan Yu
import numpy as np

def forward_algorithm(O, HMM_model):
    """HMM Forward Algorithm.
    Args:
        O: (o1, o2, ..., oT), observations
        HMM_model: (pi, A, B), (init state prob, transition prob, emitting prob)
    Return:
        prob: the probability of HMM_model generating O.
    """
    pi, A, B = HMM_model
    T = len(O)
    N = len(pi)
    prob = 0.0
    # Begin Assignment
    alpha = np.zeros((T, 1))
    for i in range(T):
        alpha[i] = pi[i]*B[i][O[0]]

    alpha_pre = alpha.copy()

    for t in range(T-1):
        for i in range(T):
            tmp = 0
            for j in range(N):
                tmp += alpha_pre[j]*A[j][i]
            alpha[i] = tmp * B[i][O[t+1]]
        alpha_pre = alpha.copy()
    for i in range(N):
        prob += alpha[i]

    # Put Your Code Here

    # End Assignment
    return prob


def backward_algorithm(O, HMM_model):
    """HMM Backward Algorithm.
    Args:
        O: (o1, o2, ..., oT), observations
        HMM_model: (pi, A, B), (init state prob, transition prob, emitting prob)
    Return:
        prob: the probability of HMM_model generating O.
    """
    pi, A, B = HMM_model
    T = len(O)
    N = len(pi)
    prob = 0.0
    # Begin Assignment
    beta = np.ones((N, 1))

    beta_t_1 = beta.copy()

    for t in range(T-2, -1, -1):
        for i in range(N):
            tmp = 0
            for j in range(N):
                tmp += A[i][j]*B[j][O[t+1]]*beta[j]
            beta_t_1[i] = tmp
        beta = beta_t_1.copy()
    for i in range(N):
        prob += pi[i]*B[i][O[0]]*beta[i]

    # Put Your Code Here

    # End Assignment
    return prob
 

def Viterbi_algorithm(O, HMM_model):
    """Viterbi decoding.
    Args:
        O: (o1, o2, ..., oT), observations
        HMM_model: (pi, A, B), (init state prob, transition prob, emitting prob)
    Returns:
        best_prob: the probability of the best state sequence
        best_path: the best state sequence
    """
    pi, A, B = HMM_model
    T = len(O)
    N = len(pi)
    best_prob, best_path = 0.0, []
    # Begin Assignment

    delta = np.zeros((T, T))
    Psi = np.zeros((T, T))
    for i in range(T):
        delta[0, i] = pi[i]*B[i][O[0]]

    for t in range(1, T):
        for i in range(T):
            tmp = np.zeros((N, 1))
            for j in range(N):
                tmp[j] = delta[t-1, j]*A[j][i]
            delta[t, i] = np.max(tmp) * B[i][O[t]]
            Psi[t, i] = np.argmax(tmp)
    best_prob = np.max(delta[-1, :])
    best_path.append(np.argmax(delta[-1, :]))
    for t in range(T-2, -1, -1):
        best_path.append(int(Psi[t+1, int(best_path[-1])]))

    # Put Your Code Here

    # End Assignment
    return best_prob, best_path


if __name__ == "__main__":
    color2id = { "RED": 0, "WHITE": 1 }
    # model parameters
    pi = [0.2, 0.4, 0.4]
    A = [[0.5, 0.2, 0.3],
         [0.3, 0.5, 0.2],
         [0.2, 0.3, 0.5]]
    B = [[0.5, 0.5],
         [0.4, 0.6],
         [0.7, 0.3]]
    # input
    observations = (0, 1, 0)
    HMM_model = (pi, A, B)
    # process
    observ_prob_forward = forward_algorithm(observations, HMM_model)
    print(observ_prob_forward)

    observ_prob_backward = backward_algorithm(observations, HMM_model)
    print(observ_prob_backward)

    best_prob, best_path = Viterbi_algorithm(observations, HMM_model) 
    print(best_prob, best_path)
