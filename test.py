import reader
import numpy as np
import sys

from tqdm import tqdm
from config import num_users, num_items

def sign_mat(RM):
	sign = np.vectorize(lambda x: 1.0 if x != 0 else 0.0)

	RM_S = sign(RM)

	return RM_S

def soft(T, s):
	softer = np.vectorize(lambda x: min(x + s, 0) if x < 0 else max(x - s, 0))

	return softer(T)

def between(l, x, r):
	return l <= x and x <= r

def near(l, x, r):
	if x <= l:
		return l

	else:
		return r

def clamp(M, l, r):
	clamper = np.vectorize(lambda x: x if between(l, x, r) else near(l, x, r))

	M_C = clamper(M)

	return M_C


reg_lambda = float(sys.args[1])
max_epochs = int(sys.args[2])

for fold in range(1, 6):
	Y = reader.get_matrix("datasets/ml-100k/u{}.base".format(fold))

	R = sign_mat(Y)

	X = np.random.rand(*Y.shape)

	X = (4 * X) + 1

	for epoch in tqdm(range(max_epochs)):
		B = X + Y - (R*X)

		U, s, V = np.linalg.svd(B, full_matrices = False)

		S = np.zeros((num_users, num_users))

		np.fill_diagonal(S, soft(s, reg_lambda))

		X = np.dot(U, np.dot(S, V))

	X = np.clip(X, 1, 5)

	T = reader.get_list("datasets/ml-100k/u{}.test".format(fold))

	err = 0

	for user, item, rating in T:
		err += np.abs(int(X[user][item]) - rating)

	MAE = err / len(T)

	NMAE = MAE / 4

	print(NMAE)