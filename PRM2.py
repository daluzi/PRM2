# _*_ coding: utf-8 _*_
# @Author   : daluzi
# @time     : 2019/10/25 10:14
# @File     : PRM2.py
# @Software : PyCharm

from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
from numpy import *
# import random_test
import copy
from sklearn.metrics import mean_squared_error #均方误差
from sklearn.metrics import mean_absolute_error #平方绝对误差
import random


# 读取txt文件
def ReadTxtData(filePath):
	resultData = []
	with open(filePath,"r") as f:
		for line in f:
			resultData.append(list(line.strip("\n").split("::")))
	# print(len(resultData))
	print(resultData)
	return resultData

#处理原始数据
def ProData(dataSet):
	dataSetSecondColu = []
	for i in range(len(dataSet)):
		dataSetSecondColu.append(dataSet[i][1])
	dataSetSecondColu = list(set(list(map(lambda x: float32(x), dataSetSecondColu))))  # 字符型转为int，然后去重
	# dataSetSecondColu = dataSetSecondColu))#去重
	dataSetSecondColu.sort()  # 排序
	print(dataSetSecondColu)
	# print("下标:\n",dataSetSecondColu.index(149532))
	user_item_matrix = np.zeros((1624, 1672))

	dataSet = [np.array(list(map(lambda x: float32(x), dataSet[i]))) for i in range(len(dataSet))]  # 字符数组转化为数字数组
	dataSet = np.array(dataSet)
	twiColu = dataSet[np.lexsort(dataSet.T[1, None])]  # 按照第二列即item排序

	print("twiColu:\n", twiColu)
	print(np.array(twiColu).shape)
	for i in range(len(dataSet)):
		m = int(dataSet[i][0])
		n = int(dataSetSecondColu.index(dataSet[i][1]))
		r = dataSet[i][2]
		# print(m,n,r)
		user_item_matrix[m][n] = r
	hang, lie = shape(user_item_matrix)
	print("hang", hang, lie)
	train_matrix = user_item_matrix
	test_matrix = np.zeros((hang, lie))
	line = np.argwhere(train_matrix > 0)
	line1 = int(0.2 * len(line))
	print(line1)

	for i in range(hang):
		randomResult = random.sample(range(1, lie), int(0.1 * lie))
		for j in range(len(randomResult)):
			o = randomResult[j]
			test_matrix[i][o] = train_matrix[i][o]
			train_matrix[i][o] = 0

	trc = np.array(np.argwhere(train_matrix > 0))
	print("trc:\n", trc)
	lentrc = len(trc)
	print("trc.length:\n", lentrc)
	# print("each:\n",trc[2][1])
	total = 0
	for i in range(lentrc):
		total += train_matrix[trc[i][0]][trc[i][1]]
	# print(total)
	trainR = total / lentrc
	print("trainR:\n", trainR)

	return trainR, train_matrix, test_matrix

# 相似矩阵
def trainW(v):
	similarMatrix = cosine_similarity(v)
	m = np.shape(similarMatrix)[0]
	for i in range(m):
		for j in range(m):
			if j == i:
				similarMatrix[i][j] = 0
	return similarMatrix

# KNN
def myKNN(S, k):
	N = len(S)  # 输出的是矩阵的行数
	A = np.zeros((N, N))

	for i in range(N):
		dist_with_index = zip(S[i], range(N))
		dist_with_index = sorted(dist_with_index, key=lambda x: x[0], reverse=True)
		# print(dist_with_index)
		neighbours_id = [dist_with_index[m][1] for m in range(k)]  # xi's k nearest neighbours
		# print("neigh",neighbours_id)
		for j in neighbours_id:  # xj is xi's neighbour
			# print(j)
			A[i][j] = 1
			A[j][i] = A[i][j]  # mutually
	# print(A[i])
	m = np.shape(A)[0]
	for i in range(m):
		for j in range(m):
			if j == i:
				A[i][j] = 0
	return A


class PRM2:

	def __init__(self,filepath,k):
		readData = ReadTxtData(filepath)#读取文件'
		r, train, test = ProData(readData)
		U, V = self.Update(train, k, r, 30, 30, 30, 0.0001)
		print("----------------------------------------------")
		print("U:\n",U)
		print("V:\n",V)
		new = np.dot(U.T, V)
		self.test = test
		self.new = new
		self.r = r
		pass

	def Update(self, R, k, r, beita, gama, yinta, l):
		'''
		:param R: user-item matrix
		:param k: The number of iterations
		:param r: r-rank factors
		:return:
		'''
		print("R:\n",R)
		I = copy.copy(R)
		I[I > 0] = 1
		print(I.shape)

		m, n = R.shape
		U = np.array(np.random.random((m, 10)),dtype='float64')
		V = np.array(np.random.random((n, 10)),dtype='float64')

		#这里可以通过KNN找到user的朋友矩阵
		simiX = trainW(R)
		W = myKNN(simiX, 5)
		print(W.shape)
		# print("w:",W)

		# updating formulas
		for i in range(k):
			R1 = np.dot(U, V.T)
			R1 = [[R1[i][j] + r for j in range(len(R1[i]))] for i in range(len(R1))]
			# U
			for i_u in range(m):
				u1 = np.zeros((1, 10))
				u5 = np.zeros((1, 10))
				u6 = np.zeros((1, 10))
				for j_u in range(n):
					if I[i_u][j_u] != 0:
						u1 = u1 + (R1[i_u][j_u] - R[i_u][j_u]) * V[j_u,:]
						u6 = u6 + np.sum(I[i_u, :]) * (np.dot(U[i_u, :], V[j_u, :].T) - simiX[i_u, j_u]) * V[j_u, :]
				# 		print(V[j_u:,].shape)
					for j_u in range(m):
						u51 = np.zeros((1, 10))
						for jj_u in range(m):
							if W[j_u][jj_u] != 0:
								u51 = u51 + simiX[j_u, jj_u] * U[jj_u,:]
						if W[i_u][j_u] != 0:
							u5 = u5 + simiX[i_u, j_u] * u51
				u51 = gama * u5

				u1 = u1 + 0.1 * U[i_u,:]
				print(u1.shape)

				u2 = np.zeros((1, 10))
				u3 = np.zeros((1, 10))
				u4 = np.zeros((1, 10))
				for j_u in range(m):
					if W[i_u][j_u-1] != 0:
						u2 = u2 + np.sum(I[j_u]) * U[j_u,:]
						u4 = u4 + simiX[i_u,j_u] * U[j_u,:]
					u31 = np.zeros((1, 10))
					for jj_u in range(m):
						if W[j_u][jj_u - 1] != 0:
							# print(I[jj_u])
							u31 = u31 + np.sum(I[jj_u]) * U[jj_u, :]
					if W[i_u][j_u] != 0:
						u3 = u3 + np.sum(I[j_u]) * u31
				u21 = beita * (U[i_u,:] - u2)
				u32 = beita * u3
				u41 = gama * (U[i_u,:] - u4)
				u61 = yinta * u6

				U[i_u,:] = U[i_u,:] - l * (u1 + u21 - u32 - u51 + u41 + u61)

			#V
			for i_v in range(n):
				v1 = np.zeros((1, 10))
				v2 = np.zeros((1, 10))
				for j_v in range(m):
					v1 = np.sum(I[j_v,:]) * (R1[i_v][j_v] - R[i_v][j_v]) * U[j_v,:]
					if I[i_v][j_v] != 0:
						v2 = v2 + np.sum(I[i_v,:]) * (np.dot(U[i_v,:], V[j_v,:].T) - simiX[i_v, j_v]) * U[j_v,:]
				v1 = v1 + 0.1 * V[i_v,:]
				v21 = yinta * v2

				V[i_v,:] = V[i_v,:] - l * (v1 + v21)
			print("run%d" % i)

		return U, V


if __name__ == '__main__':
	filePath = './Yelp/pets/ratings.txt'
	k = 10
	prm2 = PRM2(filePath, k)
	# newX = [[sr1.new[i][j] + sr1.r for j in range(len(sr1.new[i]))] for i in range(len(sr1.new))]  # 每个元素累加r
	newX = prm2.new
	xiabao = np.argwhere(prm2.test > 0)  # 获取测试集中值大于0的元素的下标
	y_true = []
	y_pred = []
	for i, j in xiabao:
		y_true.append(prm2.test[i][j])
		y_pred.append(newX[i][j])

	print("y_pred", y_pred)
	print("y_true", y_true)
	print("SR1VSS RMSE:", sqrt(mean_squared_error(y_true, y_pred)))
	print("SR1VSS MAE:", mean_absolute_error(y_true, y_pred))
