import numpy as np
from config import num_users, num_items

def get_matrix(filename):
	file = open(filename, "r")

	R = np.zeros((num_users, num_items))

	for line in file:
		user, item, rating, _ = map(int, line.split("\t"))

		R[user - 1][item - 1] = rating

	file.close()

	return R

def get_list(filename):
	file = open(filename, "r")

	A = []

	for line in file:
		user, item, rating, _ = map(int, line.split("\t"))

		A.append((user - 1, item - 1, rating))

	return A