def update_parameters_with_gd(parameters, grads, learning_rate):
	L = len(parameters)

	for l in range(L):
		parameters['W'+str(l+1)] = parameters['W'+str(l+1)] - learning_rate*grads['dW'+str(l+1)]
		parameters['b'+str(l+1)] = parameters['b'+str(l+1)] - learning_rate*grads['db'+str(l+1)]
	return parameters

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0 ):

	np.random.seed(seed)
	m = X.shape[1]
	mini_batches = []

	#shuffle
	permutation = list(wnp.random.permutation(m))
	shuffled_X = X[:, permutation]
	shuffled_Y = Y[:, permutation].reshape((1,m))

	#partition
	num_complete_minibatches = math.floor(m/mini_batch_size)
	for k in range(0, num_complete_minibatches):
		mini_batch_X = shuffled_X[:, k*mini_batch_size:(k+1)*mini_batch_size]
		mini_batch_Y = shuffled_Y[:, k*mini_batch_size:(k+1)*mini_batch_size]

		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)

	if m % mini_btch_size != 0:
		mini_batch_X = shuffled_X[:, num_complete_minibatches*mini_batch_size:(m-num_complete_minibatches*mini_batch_size)]
		mini_batch_Y = shuffled_Y[:, num_complete_minibatches*mini_batch_size:(m-num_complete_minibatches*mini_batch_size)]
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)
	return mini_batches

def model(X, Y, layers_dims, optimizer, learning_rate=0.0007, mini_batch_size=64, beta=0.9, beta1 = 0.9, beta2=0.999, epsilon=1e8, num_epochs=10000, printc_cost=True):
	



# #To do list:
#  implement all of these on my last project to see if time reduced.
# change the batch size to 1 to see the difference. 

