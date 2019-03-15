import tensorflow as tf
import math

class BBBNNRegression():

	def __init__(self, number_mini_batches, number_features, number_output_units, activation_unit, learning_rate, hidden_units, number_samples_variance_reduction, precision_alpha, weights_prior_mean_1, weights_prior_mean_2, weights_prior_deviation_1, weights_prior_deviation_2, mixture_pie, rho_mean, extra_likelihood_emphasis):
		self.number_mini_batches = tf.constant(number_mini_batches, dtype=tf.int64)
		self.activation_unit = activation_unit
		self.learning_rate = learning_rate
		self.hidden_units = hidden_units
		self.number_samples_variance_reduction = number_samples_variance_reduction
		self.precision_alpha = precision_alpha
		self.weights_prior_mean_1 = weights_prior_mean_1
		self.weights_prior_mean_2 = weights_prior_mean_2
		self.weights_prior_deviation_1 = weights_prior_deviation_1
		self.weights_prior_deviation_2 = weights_prior_deviation_2
		self.mixture_pie = mixture_pie
		self.rho_mean = rho_mean
		self.extra_likelihood_emphasis = extra_likelihood_emphasis

		self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name='global_step')
		self.mini_batch_index = tf.Variable(1, dtype=tf.int64, trainable=False, name='mini_batch_index')

		self.all_prior_cost = 0
		self.all_variational_MAP_cost = 0
		self.all_likelihood_cost = 0

		output_forward_pass_1 = None
		output_forward_pass_2 = None
		output_forward_pass_3 = None
		output_forward_pass_4 = None
		output_forward_pass_5 = None
		output_forward_pass = None
		
		# Mixture Prior
		ds = tf.contrib.distributions
		self.WEIGHTS_PRIOR = ds.Mixture(cat=ds.Categorical(probs=[self.mixture_pie, 1.- self.mixture_pie]),
									    components=[ds.Normal(loc=self.weights_prior_mean_1, scale=self.weights_prior_deviation_1), ds.Normal(loc=self.weights_prior_mean_2, scale=self.weights_prior_deviation_2)],
									    name='WEIGHTS_MIXTURE_PRIOR')

		#self.WEIGHTS_PRIOR = tf.distributions.Normal(loc=0., scale=1., name='WEIGHTS_PRIOR')

		with tf.name_scope('inputs'):
			self.X_input = tf.placeholder(tf.float32, shape=(None, number_features), name='x_input')
			self.Y_input = tf.placeholder(tf.float32, shape=(None, number_output_units), name='y_input')

		with tf.name_scope('input_output_forward_pass_mapping'):
			with tf.name_scope('between_input_and_first_hidden_layer'):
				self.sampled_weights_1, self.sampled_biases_1, variational_MAP_cost, prior_cost, mu_weights_1, rho_weights_1, mu_bias_1, rho_bias_1, sampled_weights_for_prediction, sampled_biases_for_prediction = self.getSampled_Weights_Biases_MAPCost_PriorCost(shape=(number_features, self.hidden_units[0]))
				self.all_variational_MAP_cost = self.all_variational_MAP_cost + variational_MAP_cost
				self.all_prior_cost = self.all_prior_cost + prior_cost
				for variance_reductor_iterator in range(self.number_samples_variance_reduction):
					if output_forward_pass_1 == None:
						output_forward_pass_1 = self.fetch_ACTIVATION_UNIT(tf.matmul(self.X_input, self.sampled_weights_1[variance_reductor_iterator]) + self.sampled_biases_1[variance_reductor_iterator])[None]
					else:
						output_forward_pass_1 = tf.concat([output_forward_pass_1, self.fetch_ACTIVATION_UNIT(tf.matmul(self.X_input, self.sampled_weights_1[variance_reductor_iterator]) + self.sampled_biases_1[variance_reductor_iterator])[None]], 0)
			
			with tf.name_scope('between_hidden_layers'):
				sampled_weights, sampled_biases, variational_MAP_cost, prior_cost, t1, t2,t3, t4, sampled_weights_for_prediction, sampled_biases_for_prediction = self.getSampled_Weights_Biases_MAPCost_PriorCost(shape=(self.hidden_units[0], self.hidden_units[1]))
				self.all_variational_MAP_cost = self.all_variational_MAP_cost + variational_MAP_cost
				self.all_prior_cost = self.all_prior_cost + prior_cost
				for variance_reductor_iterator in range(self.number_samples_variance_reduction):
					if output_forward_pass_2 == None:
						output_forward_pass_2 = self.fetch_ACTIVATION_UNIT(tf.matmul(output_forward_pass_1[variance_reductor_iterator], sampled_weights[variance_reductor_iterator]) + sampled_biases[variance_reductor_iterator])[None]
					else:
						output_forward_pass_2 = tf.concat([output_forward_pass_2, self.fetch_ACTIVATION_UNIT(tf.matmul(output_forward_pass_1[variance_reductor_iterator], sampled_weights[variance_reductor_iterator]) + sampled_biases[variance_reductor_iterator])[None]], 0)

		
				sampled_weights, sampled_biases, variational_MAP_cost, prior_cost, t1, t2,t3, t4, sampled_weights_for_prediction, sampled_biases_for_prediction = self.getSampled_Weights_Biases_MAPCost_PriorCost(shape=(self.hidden_units[1], self.hidden_units[2]))

				self.all_variational_MAP_cost = self.all_variational_MAP_cost + variational_MAP_cost
				self.all_prior_cost = self.all_prior_cost + prior_cost

				for variance_reductor_iterator in range(self.number_samples_variance_reduction):
					if output_forward_pass_3 == None:
						output_forward_pass_3 = self.fetch_ACTIVATION_UNIT(tf.matmul(output_forward_pass_2[variance_reductor_iterator], sampled_weights[variance_reductor_iterator]) + sampled_biases[variance_reductor_iterator])[None]
					else:
						output_forward_pass_3 = tf.concat([output_forward_pass_3, self.fetch_ACTIVATION_UNIT(tf.matmul(output_forward_pass_2[variance_reductor_iterator], sampled_weights[variance_reductor_iterator]) + sampled_biases[variance_reductor_iterator])[None]], 0)

				
				'''		
				sampled_weights, sampled_biases, variational_MAP_cost, prior_cost, t1, t2,t3, t4, sampled_weights_for_prediction, sampled_biases_for_prediction = self.getSampled_Weights_Biases_MAPCost_PriorCost(shape=(self.hidden_units[2], self.hidden_units[3]))

				self.all_variational_MAP_cost = self.all_variational_MAP_cost + variational_MAP_cost
				self.all_prior_cost = self.all_prior_cost + prior_cost

				for variance_reductor_iterator in range(self.number_samples_variance_reduction):

					if output_forward_pass_4 == None:
						output_forward_pass_4 = self.fetch_ACTIVATION_UNIT(tf.matmul(output_forward_pass_3[variance_reductor_iterator], sampled_weights[variance_reductor_iterator]) + sampled_biases[variance_reductor_iterator])[None]
						output_validation_4 = self.fetch_ACTIVATION_UNIT(tf.matmul(output_validation_3[variance_reductor_iterator], sampled_weights_for_prediction[variance_reductor_iterator]) + sampled_biases_for_prediction[variance_reductor_iterator])[None]
					else:
						output_forward_pass_4 = tf.concat([output_forward_pass_4, self.fetch_ACTIVATION_UNIT(tf.matmul(output_forward_pass_3[variance_reductor_iterator], sampled_weights[variance_reductor_iterator]) + sampled_biases[variance_reductor_iterator])[None]], 0)
						output_validation_4 = tf.concat([output_validation_4, self.fetch_ACTIVATION_UNIT(tf.matmul(output_validation_3[variance_reductor_iterator], sampled_weights_for_prediction[variance_reductor_iterator]) + sampled_biases_for_prediction[variance_reductor_iterator])[None]], 0)



				sampled_weights, sampled_biases, variational_MAP_cost, prior_cost, t1, t2,t3, t4, sampled_weights_for_prediction, sampled_biases_for_prediction = self.getSampled_Weights_Biases_MAPCost_PriorCost(shape=(self.hidden_units[3], self.hidden_units[4]))

				self.all_variational_MAP_cost = self.all_variational_MAP_cost + variational_MAP_cost
				self.all_prior_cost = self.all_prior_cost + prior_cost

				for variance_reductor_iterator in range(self.number_samples_variance_reduction):

					if output_forward_pass_5 == None:
						output_forward_pass_5 = self.fetch_ACTIVATION_UNIT(tf.matmul(output_forward_pass_4[variance_reductor_iterator], sampled_weights[variance_reductor_iterator]) + sampled_biases[variance_reductor_iterator])[None]
						output_validation_5 = self.fetch_ACTIVATION_UNIT(tf.matmul(output_validation_4[variance_reductor_iterator], sampled_weights_for_prediction[variance_reductor_iterator]) + sampled_biases_for_prediction[variance_reductor_iterator])[None]
					else:
						output_forward_pass_5 = tf.concat([output_forward_pass_5, self.fetch_ACTIVATION_UNIT(tf.matmul(output_forward_pass_4[variance_reductor_iterator], sampled_weights[variance_reductor_iterator]) + sampled_biases[variance_reductor_iterator])[None]], 0)
						output_validation_5 = tf.concat([output_validation_5, self.fetch_ACTIVATION_UNIT(tf.matmul(output_validation_4[variance_reductor_iterator], sampled_weights_for_prediction[variance_reductor_iterator]) + sampled_biases_for_prediction[variance_reductor_iterator])[None]], 0)



			with tf.name_scope('Between_Last_Hidden_and_Output_Layer'):

				sampled_weights, sampled_biases, variational_MAP_cost, prior_cost, mu_weights_2, rho_weights_2, mu_bias_2, rho_bias_2, sampled_weights_for_prediction, sampled_biases_for_prediction = self.getSampled_Weights_Biases_MAPCost_PriorCost(shape=(self.hidden_units[-1], number_output_units))

				self.all_variational_MAP_cost = self.all_variational_MAP_cost + variational_MAP_cost
				self.all_prior_cost = self.all_prior_cost + prior_cost


				for variance_reductor_iterator in range(self.number_samples_variance_reduction):

					if output_forward_pass == None:
						output_forward_pass = (tf.matmul(output_forward_pass_5[variance_reductor_iterator], sampled_weights[variance_reductor_iterator]) + sampled_biases[variance_reductor_iterator])[None]
						output_validation = (tf.matmul(output_validation_5[variance_reductor_iterator], sampled_weights_for_prediction[variance_reductor_iterator]) + sampled_biases_for_prediction[variance_reductor_iterator])[None]
					else:
						output_forward_pass = tf.concat([output_forward_pass, (tf.matmul(output_forward_pass_5[variance_reductor_iterator], sampled_weights[variance_reductor_iterator]) + sampled_biases[variance_reductor_iterator])[None]], 0)
						output_validation = tf.concat([output_validation, (tf.matmul(output_validation_5[variance_reductor_iterator], sampled_weights_for_prediction[variance_reductor_iterator]) + sampled_biases_for_prediction[variance_reductor_iterator])[None]], 0)

			'''

			with tf.name_scope('between_last_hidden_and_output_layer'):
				sampled_weights, sampled_biases, variational_MAP_cost, prior_cost, mu_weights_2, rho_weights_2, mu_bias_2, rho_bias_2, sampled_weights_for_prediction, sampled_biases_for_prediction = self.getSampled_Weights_Biases_MAPCost_PriorCost(shape=(self.hidden_units[-1], number_output_units))
				self.all_variational_MAP_cost = self.all_variational_MAP_cost + variational_MAP_cost
				self.all_prior_cost = self.all_prior_cost + prior_cost
				for variance_reductor_iterator in range(self.number_samples_variance_reduction):
					if output_forward_pass == None:
						output_forward_pass = (tf.matmul(output_forward_pass_3[variance_reductor_iterator], sampled_weights[variance_reductor_iterator]) + sampled_biases[variance_reductor_iterator])[None]
					else:
						output_forward_pass = tf.concat([output_forward_pass, (tf.matmul(output_forward_pass_3[variance_reductor_iterator], sampled_weights[variance_reductor_iterator]) + sampled_biases[variance_reductor_iterator])[None]], 0)
					model_distribution = tf.distributions.Normal(loc=output_forward_pass[variance_reductor_iterator], scale=(1.0/tf.sqrt(self.precision_alpha)))
					self.all_likelihood_cost = self.all_likelihood_cost + tf.reduce_sum(model_distribution.log_prob(self.Y_input))

		with tf.name_scope('final_outputs'):
			mean_of_output_forward_pass_temporary, variance_of_output_forward_pass = tf.nn.moments(output_forward_pass, 0, name='prediction_mean_n_variance')
			self.mean_of_output_forward_pass = tf.identity(mean_of_output_forward_pass_temporary, name='prediction_mean')
			self.deviation_of_output_forward_pass = tf.sqrt(variance_of_output_forward_pass, name='prediction_standard_deviation')
			self.maximum_of_output_forward_pass = tf.reduce_max(output_forward_pass, axis=0, name='prediction_maximum')
			self.minimum_of_output_forward_pass = tf.reduce_min(output_forward_pass, axis=0, name='prediction_minimum')

		with tf.name_scope('cost'):
			intercost_minibatch_weight_pie = (tf.pow(2., tf.to_float(self.number_mini_batches - self.mini_batch_index)))/(tf.pow(2., tf.to_float(self.number_mini_batches)) - 1)
			self.complexity_cost = self.all_variational_MAP_cost - self.all_prior_cost
			self.cost = tf.subtract((intercost_minibatch_weight_pie * (self.all_variational_MAP_cost - self.all_prior_cost)), (self.extra_likelihood_emphasis * self.all_likelihood_cost), name='cost')

		with tf.name_scope('error'):
			self.mean_squared_error = tf.reduce_mean(tf.squared_difference(self.mean_of_output_forward_pass, self.Y_input), name='mean_squared_error')

		with tf.name_scope('optimization'):
			optimizer = tf.train.AdamOptimizer(self.learning_rate, name='adam_optimizer')
			self.training = optimizer.minimize(self.cost, global_step=self.global_step, name='training')

		with tf.name_scope('summaries'):
			tf.summary.scalar(name='mean_squared_error_log', tensor=self.mean_squared_error)
			tf.summary.scalar(name='deviation_of_output_forward_pass_log', tensor=tf.reduce_mean(self.deviation_of_output_forward_pass))
			tf.summary.scalar(name='prior_cost_log', tensor=self.all_prior_cost)
			tf.summary.scalar(name='variational_MAP_cost_log', tensor=self.all_variational_MAP_cost)
			tf.summary.scalar(name='likelihood_cost_log', tensor=self.all_likelihood_cost)
			tf.summary.scalar(name='complexity_cost_log', tensor=self.complexity_cost)
			tf.summary.scalar(name='cost_log', tensor=self.cost)
			self.summary_op = tf.summary.merge_all()

		with tf.name_scope('mini_batch_index_update'):
			self.mini_batch_index_update = tf.assign(ref=self.mini_batch_index, value=((self.mini_batch_index % self.number_mini_batches) + 1), name='mini_batch_index_update')


	def fetch_ACTIVATION_UNIT(self, param):
		if self.activation_unit == 'RELU':
			return tf.nn.relu(param)
		elif self.activation_unit == 'SIGMOID':
			return tf.sigmoid(param)
		elif self.activation_unit == 'TANH':
			return tf.tanh(param)


	def update_mini_batch_index(self):
		return self.mini_batch_index_update


	def getCostforTraining(self):
		return self.all_prior_cost.eval(), self.all_variational_MAP_cost.eval() , self.all_likelihood_cost.eval(), self.cost.eval()
		#return self.likelihood_cost, self.cost


	def summarize(self):
		return self.summary_op


	def makeInference(self):
		return self.mean_of_output_forward_pass, self.deviation_of_output_forward_pass, self.maximum_of_output_forward_pass, self.minimum_of_output_forward_pass


	def train(self):
		return self.training


	def getMeanSquaredError(self):
		return self.mean_squared_error


	def getWeightsAndBiases(self):
		return self.sampled_weights_1


	def getSampled_Weights_Biases_MAPCost_PriorCost(self, shape):

		mu_weights = tf.Variable(tf.zeros(shape=[self.number_samples_variance_reduction, shape[0], shape[1]], dtype=tf.float32), name='mu_weights')
		rho_weights = tf.Variable(tf.truncated_normal(shape=[self.number_samples_variance_reduction, shape[0], shape[1]], mean=self.rho_mean, stddev = 1.0/(math.sqrt(float(shape[0]))), dtype=tf.float32), name='rho_weights')

		mu_bias = tf.Variable(tf.zeros(shape=[self.number_samples_variance_reduction, shape[1]], dtype=tf.float32), name='mu_bias')
		rho_bias = tf.Variable(tf.truncated_normal(shape=[self.number_samples_variance_reduction, shape[1]], mean=self.rho_mean, stddev = 1., dtype=tf.float32), name='rho_bias')

		sampled_weights = mu_weights + ( tf.log(1 + tf.exp( rho_weights)) * tf.random_normal(shape=[self.number_samples_variance_reduction, shape[0], shape[1]], mean=0., stddev=1.0, name='weights_randomizer'))
		sampled_biases = mu_bias + ( tf.log(1 + tf.exp(rho_bias)) * tf.random_normal(shape=[self.number_samples_variance_reduction, shape[1]], mean=0., stddev= 1.0, name='bias_randomizer'))

		sampled_weights_for_prediction = mu_weights + ( tf.log(1 + tf.exp(rho_weights)) * tf.random_normal(shape=[self.number_samples_variance_reduction, shape[0], shape[1]], mean=0., stddev=1.0, name='weights_randomizer_for_prediction'))
		sampled_biases_for_prediction = mu_bias + ( tf.log(1 + tf.exp(rho_bias)) * tf.random_normal(shape=[self.number_samples_variance_reduction, shape[1]], mean=0., stddev= 1.0, name='bias_randomizer_for_prediction'))

		variational_distribution_weights = tf.distributions.Normal(loc=mu_weights, scale=tf.log(1 + tf.exp(rho_weights)))
		variational_distribution_bias = tf.distributions.Normal(loc=mu_bias, scale=tf.log(1 + tf.exp(rho_bias)))

		all_variational_MAP_cost = tf.reduce_sum(variational_distribution_weights.log_prob(sampled_weights)) + tf.reduce_sum(variational_distribution_bias.log_prob(sampled_biases))

		all_prior_cost = tf.reduce_sum(self.WEIGHTS_PRIOR.log_prob(sampled_weights)) + tf.reduce_sum(self.WEIGHTS_PRIOR.log_prob(sampled_biases))
		
		return sampled_weights, sampled_biases, all_variational_MAP_cost, all_prior_cost, mu_weights, rho_weights, mu_bias, rho_bias, sampled_weights_for_prediction, sampled_biases_for_prediction