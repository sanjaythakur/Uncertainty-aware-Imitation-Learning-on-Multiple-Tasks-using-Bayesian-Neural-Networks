import numpy as np

class Sliding_Block():
	def __init__(self, mass, initial_state=np.array([[0., 0.]])):
		delta_time = 0.1
		self.mass = mass

		self.A = np.array([[1., 1.],[0, 1.]])
		self.B = np.array([[0.],[delta_time/self.mass]])
		self.Q = np.array([[10., 0.],[0, 1.]])
		self.R = np.array([[0.1]])

		#self.state = 10. * np.random.randn(2,1)
		#self.state = np.array([[10.], [10.]])
		self.state = np.reshape(initial_state, (-1, 1))
		self.goal_state = np.array([[0.],[0.]])
		interval = np.array([[.5],[.5]])
		self.goal_state_lower = np.subtract(self.goal_state, interval)
		self.goal_state_higher = np.add(self.goal_state, interval)

	def step(self, u):
		self.state = np.dot(self.A, self.state) + np.multiply(self.B, u)
		#self.cost = np.sum(np.absolute(np.subtract(self.state, self.goal_state)))
		self.cost = np.dot(np.dot(self.state.T, self.Q), self.state) + np.dot(np.dot(u.T, self.R), u)
		if (self.goal_state_lower <= self.state).all() and (self.state <= self.goal_state_higher).all():
			finish = True
		else:
			finish = False
		return self.state, self.cost, finish

	def reset(self):
		self.state = 10. * np.random.randn(2,1)