import numpy as np
from matplotlib import pyplot as plt, cm
import plotly.graph_objects as go
import matplotlib.patches as patches
# from mpl_toolkits.mplot3d import axes3d
# import kaleido
class GKLS:
	def __init__(self, N: int = 2, m: int = 5, v: float = -1,
				 r: float = 0.4, d: float = 0.8):
		super().__init__()
		self.N = N #problem dimension
		self.m = m #number of local minima
		self.v = v #value of the global minimum
		self.r = r #radius of the attraction region of the global minimizer
		self.d = d #distance of the paraboloid vertex to the global minimizer
		self.setup()

	def setup(self):
		self.t = 0 #paraboloid minimum
		self.T = np.zeros((self.N)) #paraboloid minimizers
		self.f = np.zeros(self.m) #local minima
		self.M = np.zeros((self.m, self.N)) #local minimizers
		self.R = np.zeros(self.m) #radii of attraction regions
		self.B = np.array([-1, 1]) #box constraints
		self.peak = np.zeros(self.m)
		self.gap = self.r #gap between local minimizers
		self.e = 1e-10 #error precision
		self.generate()

	def generate(self):
		for i in range(self.N):
			#randomize paraboloid minimizer coordinates and the paraboloid minimum
			self.M[0][i] = self.B[0] + np.random.rand() * (self.B[1] - self.B[0])
		self.f[0] = self.t #fix the paraboloid minimum
		self.T = self.M[0]
		#generate the global minimizer based on generalized spherical coordinates
		#using arbitrary vector phi and distance from the paraboloid vertex

		#generate an angle 0 <= phi(0) <= PI and the coordinate x(0)
		self.rand = np.random.rand()
		self.M[1][0] = self.M[0][0] + self.d * np.cos(np.pi * self.rand)
		if (self.M[1][0] < self.B[0] or self.M[1][0] > self.B[1]):
			self.M[1][0] = self.M[0][0] - self.d * np.cos(np.pi * self.rand)
		self.sin_phi = np.sin(np.pi * self.rand)

		#generate remaining angles 0 <= phi(i) <= 2 * PI and the coordinates x(i), i = 1, 2, ..., N - 2
		self.rand = np.random.rand()
		for i in range(1, self.N - 1):
			self.M[1][i] = self.M[0][i] + self.d * np.cos(2.0 * np.pi * self.rand) * self.sin_phi
			if (self.M[1][i] < self.B[0] or self.M[1][i] > self.B[1]):
				self.M[1][i] = self.M[0][i] - self.d * np.cos(2.0 * np.pi * self.rand) * self.sin_phi
			self.sin_phi *= np.sin(np.pi * self.rand)

		#generate the last coordinate
		self.M[1][self.N - 1] = self.M[0][self.N - 1] + self.d * self.sin_phi 
		if (self.M[1][self.N - 1] < self.B[0] or self.M[1][self.N - 1] > self.B[1]):
			self.M[1][self.N - 1] = self.M[0][self.N - 1] - self.d * self.sin_phi
		self.f[1] = self.v#set the global minimum value

		self.w = np.full(self.m, 0.99) #set the weight coefficients
		self.w[1] = 1.0

		while (True):
			i = 2
			while(i < self.m):
				while(True):
					for j in range(self.N):
						self.M[i][j] = self.B[0] + np.random.rand() * (self.B[1] - self.B[0])  
					if ((self.r + self.gap) - np.linalg.norm(self.M[i] - self.M[1]) < self.e):
						break
				i += 1
			if (self.coincidence_check()): break
		self.set_basins()

	def coincidence_check(self):
		for j in range(2, self.m):
			self.temp_d = np.linalg.norm(self.M[j] - self.M[0])
			if (self.temp_d < self.e): #too close, space out more
				return False
		for j in range(1, self.m - 1):
			for k in range(j + 1, self.m):
				self.temp_d = np.linalg.norm(self.M[j] - self.M[k])
				if (self.temp_d < self.e):
					return False
		return True

	def set_basins(self):
		#set other local minimizers randomly while satisfying constraints
		for i in range(self.m):
			self.temp_min = 1e100
			for j in range(self.m):
				if (i == j): continue
				self.temp_d = np.linalg.norm(self.M[i] - self.M[j])
				if (self.temp_d < self.temp_min):
					self.temp_min = self.temp_d 
			self.R[i] = self.temp_min / 2.0

		self.R[1] = self.r

		for i in range(2, self.m):
			self.temp_d = np.linalg.norm(self.M[i] - self.M[1])
			if (self.temp_d - self.R[1] - self.e < self.R[i]):
				self.R[i] = self.temp_d

#good

		for i in range(self.m):
			if (i == 1): continue
			self.temp_min = 1e100
			for j in range(self.m):
				if (i == j): continue
				self.temp_d = np.linalg.norm(self.M[i] - self.M[j]) - self.R[j]
				if (self.temp_d  < self.temp_min):
					self.temp_min = self.temp_d
			if (self.temp_min > self.R[i] + self.e):
				self.R[i] = self.temp_min

		for i in range(self.m):
			self.R[i] *= self.w[i]

		for i in range(2, self.m):
			self.temp_d1 = np.linalg.norm(self.M[0] - self.M[i])
			self.temp_min = (self.R[i] - self.temp_d1)**2 + self.f[0] #conditional minimum at boundary
			self.rand = np.random.rand()
			self.temp_d1 = (1.0 + self.rand) * self.R[i]
			self.temp_d2 = self.rand * (self.temp_min - self.v)
			self.temp_d1 = min(self.temp_d1, self.temp_d2)
			self.peak[i] = self.temp_d1
			self.f[i] = self.temp_min - self.peak[i]

		self.gm = 0 #number of global minima
		self.gm_index = np.zeros(self.m)
		for i in range(self.m):
			if (self.f[i] >= self.v - self.e and self.f[i] <= self.v + self.e):
				self.gm_index[self.gm] = i #global minimizer index
				self.gm += 1
			else:
				self.gm_index[self.m - 1 - i + self.gm] = i #local minimizer index

		if (self.gm == 0):
			raise Exception("Global Minimizer not found")

	# def D(self, x):
	# 	for i in range(self.N):
	# 		if (x[i] < self.B[0] - self.e or x[i] > self.B[1] + self.e):
	# 			return 1e100
	# 	i = 1
	# 	while (i < self.m and np.linalg.norm(self.M[i] - x) > self.R[i]):
	# 		i += 1
	# 	if (i == self.m):
	# 		norm = np.linalg.norm(self.M[0] - x)
	# 		return norm**2 + self.f[0] #value of paraboloid function
	# 	if (np.linalg.norm(self.M[i] - x) < self.e):
	# 		return self.f[i]
	# 	norm = np.linalg.norm(self.M[0] - self.M[i])
	# 	A = norm**2 + self.f[0] - self.f[i]
	# 	norm = np.linalg.norm(self.M[i] - x)
	# 	dot = np.dot(x - self.M[i], self.M[0] - self.M[i])
	# 	return ((2 * dot) / (self.R[i]**2 * norm) - 2 * A / self.R[i]**3) * norm**3 + \
	# 		   (1 - (4 * dot) / (self.R[i] * norm) + 3 * A / self.R[i]**2) * norm**2 + self.f[i]

	def evaluate(self, x):
	    for i in range(self.N):
	        if (x[i] < self.B[0] - self.e or x[i] > self.B[1] + self.e):
	            return 1e100
	    i = 1
	    while (i < self.m and np.linalg.norm(self.M[i] - x) > self.R[i]):
	        i += 1
	    if (i == self.m):
	        norm = np.linalg.norm(self.M[0] - x)
	        return norm**2 + self.f[0]  # value of paraboloid function
	    if (np.linalg.norm(self.M[i] - x) < self.e):
	        return self.f[i]
	    norm = np.linalg.norm(self.M[0] - self.M[i])
	    A = norm**2 + self.f[0] - self.f[i]
	    norm = np.linalg.norm(self.M[i] - x)
	    dot = np.dot(x - self.M[i], self.M[0] - self.M[i])
	    return ((2 * dot) / (self.R[i]**2 * norm) - 2 * A / self.R[i]**3) * norm**3 + \
	           (1 - (4 * dot) / (self.R[i] * norm) + 3 * A / self.R[i]**2) * norm**2 + self.f[i]
