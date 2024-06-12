import numpy as np


def softmax(h_b):
    z = h_b - np.max(h_b)
    return np.exp(z) / np.sum(np.exp(z))


def greedy(q):
    return np.random.choice(np.flatnonzero(q == np.max(q)))


def eps_greedy(q, num_actions, epsilon=0.05):
    if np.random.random() < epsilon:
        idx = np.random.randint(num_actions)
    else:
        idx = greedy(q)
    return idx


def ucb(q, c, step, N):
    ucb_eq = q + c * np.sqrt(np.log(step) / N)
    return greedy(ucb_eq)

def Boltzmann(q, t=0.4):
    return np.exp(q / t) / np.sum(np.exp(q / t))