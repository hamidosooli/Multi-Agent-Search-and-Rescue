from time import ctime, time

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np
from numba import njit, prange
from scipy.sparse.csgraph import laplacian
import networkx as nx

from graph_optimization import fdla_weights_symmetric, fmmc_weights, lmsc_weights, fastest_averaging_constant_weight, max_degree_weights, metropolis_hastings_weights


def main():
	N = 10
	runs = 1000
	T = 1000

	# adjacency matrices
	As = [
		# all-to-all
		np.array([
			[0, 1, 1, 1, 1],
			[1, 0, 1, 1, 1],
			[1, 1, 0, 1, 1],
			[1, 1, 1, 0, 1],
			[1, 1, 1, 1, 0],
		]),
		# house
		np.array([
			[0, 1, 1, 0, 0],
			[1, 0, 1, 1, 0],
			[1, 1, 0, 0, 1],
			[0, 1, 0, 0, 1],
			[0, 0, 1, 1, 0],
		]),
		# ring
		np.array([
			[0, 1, 0, 0, 1],
			[1, 0, 1, 0, 0],
			[0, 1, 0, 1, 0],
			[0, 0, 1, 0, 1],
			[1, 0, 0, 1, 0],
		]),
		# line
		np.array([
			[0, 1, 0, 0, 0],
			[1, 0, 1, 0, 0],
			[0, 1, 0, 1, 0],
			[0, 0, 1, 0, 1],
			[0, 0, 0, 1, 0],
		]),
		# star
		np.array([
			[0, 1, 1, 1, 1],
			[1, 0, 0, 0, 0],
			[1, 0, 0, 0, 0],
			[1, 0, 0, 0, 0],
			[1, 0, 0, 0, 0],
		]),
		# 8-agents
		np.array([
			[0, 1, 1, 1, 0, 0, 0, 0],
			[1, 0, 1, 0, 1, 0, 0, 0],
			[1, 1, 0, 1, 1, 0, 0, 0],
			[1, 0, 1, 0, 1, 1, 1, 1],
			[0, 1, 1, 1, 0, 1, 1, 1],
			[0, 0, 0, 1, 1, 0, 1, 1],
			[0, 0, 0, 1, 1, 1, 0, 1],
			[0, 0, 0, 1, 1, 1, 1, 0],
		]),
	]

	# corresponding incidence matrices
	Is = [
		# all-to-all
		np.array([
			[ 1,  1,  1,  1,  0,  0,  0,  0,  0,  0],
			[-1,  0,  0,  0,  1,  1,  1,  0,  0,  0],
			[ 0, -1,  0,  0, -1,  0,  0,  1,  1,  0],
			[ 0,  0, -1,  0,  0, -1,  0, -1,  0,  1],
			[ 0,  0,  0, -1,  0,  0, -1,  0, -1, -1],
		]),
		# house
		np.array([
			[ 1,  0,  0,  0, -1,  0],
			[ 0,  0,  0, -1,  1,  1],
			[-1,  1,  0,  0,  0, -1],
			[ 0,  0, -1,  1,  0,  0],
			[ 0, -1,  1,  0,  0,  0],
		]),
		# ring
		np.array([
			[ 1,  0,  0,  0, -1],
			[-1,  1,  0,  0,  0],
			[ 0, -1,  1,  0,  0],
			[ 0,  0, -1,  1,  0],
			[ 0,  0,  0, -1,  1],
		]),
		# line
		np.array([
			[ 1,  0,  0,  0],
			[-1,  1,  0,  0],
			[ 0, -1,  1,  0],
			[ 0,  0, -1,  1],
			[ 0,  0,  0, -1],
		]),
		# star
		np.array([
			[ 1,  1,  1,  1],
			[-1,  0,  0,  0],
			[ 0, -1,  0,  0],
			[ 0,  0, -1,  0],
			[ 0,  0,  0, -1],
		]),
		# 8-agents
		np.array([
			[  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
			[ -1,  0,  0,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
			[  0, -1,  0, -1,  0,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
			[  0,  0, -1,  0,  0, -1,  0,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0],
			[  0,  0,  0,  0, -1,  0, -1, -1,  0,  0,  0,  1,  1,  1,  0,  0,  0],
			[  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0, -1,  0,  0,  1,  1,  0],
			[  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0, -1,  0, -1,  0,  1],
			[  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0, -1,  0, -1, -1],
		]),
	]

	networks = [
		'all-to-all',
		'house',
		'ring',
		'line',
		'star',
		'8-agents',
	]

	trueMeans = np.array([np.random.normal(0, 1, N) for _ in range(runs)])

	# sizing
	SMALL_SIZE = 10
	MEDIUM_SIZE = 14
	LARGE_SIZE = 18

	# plt.rcParams["figure.figsize"] = (15, 8)
	plt.rc('font', size=SMALL_SIZE)
	plt.rc('axes', titlesize=MEDIUM_SIZE, labelsize=MEDIUM_SIZE)
	plt.rc('xtick', labelsize=MEDIUM_SIZE)
	plt.rc('ytick', labelsize=MEDIUM_SIZE)

	np.set_printoptions(threshold=99999999999999999, linewidth=99999999999999999)

	# cm = plt.get_cmap('gist_rainbow')
	colors = [
		'tab:blue',
		'tab:orange',
		'tab:green',
		'tab:red',
		'tab:purple',
		'tab:brown',
		'tab:pink',
		'tab:gray',
		'tab:olive',
		'tab:cyan',
	]

	markers = [
		'o',
		'^',
		's',
		'x',
		'v',
		'*',
		'1',
		'D',
		'P',
	]

	print(f'Simulation started at {ctime(time())}')

	for mat_idx in range(len(As)):
		print(f'Network: {networks[mat_idx]}')
		Ps, rhos, labels = [], [], []

		for k in [0.02]:
			P, rho = generateP(As[mat_idx], kappa=k)
			Ps.append(P)
			rhos.append(rho)
			print(f'{"kappa " + str(k):<20s}: {rho} {(1 / np.log(1 / rho))}')
			labels.append(fr'$\kappa$ = {k}')
			# print(P)

		# constant edge
		alpha, _, P, rho = fastest_averaging_constant_weight(Is[mat_idx])
		rho = get_rho(P)
		Ps.append(P)
		rhos.append(rho)
		print(f'{"Constant-edge":<20s}: {rho} {(1 / np.log(1 / rho))}')
		labels.append(fr'Constant-edge ($\alpha$ = {alpha})')
		# print(P)

		# maximum degree
		alpha, _, P, rho = max_degree_weights(Is[mat_idx])
		rho = get_rho(P)
		Ps.append(P)
		rhos.append(rho)
		print(f'{"Max-degree":<20s}: {rho} {(1 / np.log(1 / rho))}')
		labels.append(fr'Maximum-degree ($\alpha$ = {alpha})')
		# print(P)

		# local degree (MH)
		_, P, rho = metropolis_hastings_weights(Is[mat_idx])
		rho = get_rho(P)
		Ps.append(P)
		rhos.append(rho)
		print(f'{"Local-degree":<20s}: {rho} {(1 / np.log(1 / rho))}')
		labels.append(fr'Local-degree')
		# print(P)

		# fmmc
		_, P, rho = fmmc_weights(Is[mat_idx])
		rho = get_rho(P)
		Ps.append(P)
		rhos.append(rho)
		print(f'{"FMMC":<20s}: {rho} {(1 / np.log(1 / rho))}')
		labels.append('FMMC')
		# print(P)

		# fdla
		_, P, rho = fdla_weights_symmetric(Is[mat_idx])
		rho = get_rho(P)
		Ps.append(P)
		rhos.append(rho)
		print(f'{"FDLA":<20s}: {rho} {(1 / np.log(1 / rho))}')
		labels.append('FDLA')

		# lmsc
		_, P = lmsc_weights(Is[mat_idx])
		rho = get_rho(P)
		Ps.append(P)
		rhos.append(rho)
		print(f'{"LMSC":<20s}: {rho} {(1 / np.log(1 / rho))}')
		labels.append('LMSC')
		print(P)
		print('\n\n')

		fig, ax = plt.subplots()

		for idx, P in enumerate(Ps):
			reg = run(runs, N, T, trueMeans, P)
			reg = np.mean(reg, axis=0)	# mean over runs

			np.save(f'newtestdata/new_{networks[mat_idx].replace(" ", "-")}_reg_{labels[idx]}.npy', reg)
			# reg = np.load(f'testdata/new_{networks[mat_idx].replace(" ", "-")}_reg_{labels[idx]}.npy')

			# plot regret
			fig.suptitle(f'{networks[mat_idx].title()} network')
			fig.supxlabel('Timesteps')
			fig.supylabel('Mean Cumulative Regret')
			ax.plot(np.cumsum(np.mean(reg, axis=0)), marker=markers[idx], markevery=200, lw=2, linestyle='solid', color=colors[idx], label=labels[idx])

		ax.grid(True)
		ax.legend()
		plt.savefig(f'newtestimg/new_{networks[mat_idx].replace(" ", "-")}-reg.svg', format='svg', bbox_inches='tight')
		plt.savefig(f'newtestimg/new_{networks[mat_idx].replace(" ", "-")}-reg.png', format='png', bbox_inches='tight')
		fig, ax = plt.subplots()

		fig.suptitle(f'{networks[mat_idx].title()} network')
		fig.supxlabel('Timesteps')
		fig.supylabel('Mean estimate error for the best arm')


@njit(parallel=True)
def run(runs: int, N: int, T: int, trueMeans: np.ndarray, P: np.ndarray, noisy_agents=np.array([0])) -> tuple:
	'''
	Plays coopucb2 given the number of runs, number of arms, timesteps, true
	means of arms, and the P matrix of the network. Optimized to work with
	numba.
	'''
	sigma_g = 1		# try 10
	eta = 2		# try 2, 2.2, 3.2
	gamma = 2.9 	# try 1.9, 2.9
	f = lambda t : np.sqrt(np.log(t))
	Geta = 1 - (eta ** 2)/16
	var = 1		# variance for the gaussian distribution behind each arm
	M, _ = P.shape
	x1 = 2 * gamma / Geta

	reg = np.zeros((runs, M, T))

	# run coop-ucb2 "runs" number of times
	for run in prange(runs):
		Q = np.zeros((M, N))	# estimated reward
		n = np.zeros((M, N))	# number of times an arm has been selected by each agent
		s = np.zeros((M, N))	# cumulative expected reward
		xsi = np.zeros((M, N))	# number of times that arm has been selected in that timestep
		rew = np.zeros((M, N))	# reward
		bestArm = np.max(trueMeans[run])
		bestArmIdx = np.argmax(trueMeans[run])

		for t in range(T):
			_t = t - 1 if t > 0 else 0
			if t < N:
				for k in range(M):
					rew[k] = np.zeros(N)
					xsi[k] = np.zeros(N)
					action = t

					rew[k, action] = np.random.normal(trueMeans[run, action], var)
					reg[run, k, t] = bestArm - trueMeans[run, action]
					xsi[k, action] += 1
			else:
				for k in range(M):
					for i in range(N):
						x0 = s[k, i] / n[k, i]
						x2 = (n[k, i] + f(_t)) / (M * n[k, i])
						x3 = np.log(_t) / n[k, i]
						_explr = sigma_g * np.sqrt(x1 * x2 * x3)
						Q[k, i] = x0 + _explr

					rew[k] = np.zeros(N)
					xsi[k] = np.zeros(N)

					action = np.argmax(Q[k, :])
					rew[k, action] = np.random.normal(trueMeans[run, action], var)
					reg[run, k, t] = bestArm - trueMeans[run, action]
					xsi[k, action] += 1

			for k in noisy_agents:
				# add noise
				rew[k, :] *= -1 * np.abs(np.random.normal(0, 1))

			# update estimates using running consensus
			for i in range(N):
				n[:, i] = P @ (n[:, i] + xsi[:, i])
				s[:, i] = P @ (s[:, i] + rew[:, i])

	return reg

def generateP(A, kappa):
	dmax = np.max(np.sum(A, axis=0))
	L = laplacian(A, normed=False)
	M, _ = np.shape(A)
	I = np.eye(M)

	P = I - (kappa/dmax) * L

	return P, get_rho(P)

def get_rho(P):
	n = P.shape[0]
	_P = P - np.ones((n, n)) * (1/n)
	l = np.abs(np.linalg.eigvals(_P))
	l = l[1 - l > 1e-5]
	return np.max(l)

if __name__ == '__main__':
	main()
