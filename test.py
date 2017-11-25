import reader
import numpy as np

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


reg_lambda = 0.1
max_epochs = 10

for fold in range(1, 6):
	Y = reader.get_matrix("datasets/ml-100k/u{}.base".format(fold))

	R = sign_mat(Y)

	X = np.random.rand(*Y.shape)

	for epoch in range(max_epochs):
		U, s, V = np.linalg.svd(X)

		S = np.zeros((num_users, num_items))

		np.fill_diagonal(S, s)

		X = np.dot(U, np.dot(S, V))

	X = np.clip(X, 1, 5)

	T = reader.get_list("datasets/ml-100k/u{}.test".format(fold))

	err = 0

	for user, item, rating in T:
		err += np.abs(int(X[user][item]) - rating)

	MAE = err / len(T)

	NMAE = MAE / 4

	print(NMAE)