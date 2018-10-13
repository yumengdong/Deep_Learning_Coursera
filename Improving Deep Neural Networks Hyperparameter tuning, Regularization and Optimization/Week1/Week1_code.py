def model(X, Y, learning_rate = 0.01, num_iterations = 1500, print_cost=True, initialization='he'):
	layers_dims = [X.shape[0],10,5,1]

	for i in range(0, num_iterations):
		if print_cost and i%1000 == 0:
			print("Cost after iteration {}: {}".format(i,cost))
			costs.append(cost)

parameters['W'+str(l)] = np.random.randn()

def model(X, Y, learning_rate = 0.3, num_iterations = 30000, print_cost=True, lambd=0, keep_prob=1):
	for i in range(0,num_iterations):

numerator = np.linalg.norm()

def gradient_check(x, theta, epsilon=1e-7):
	thetaplus = theta + epsilon
	thetaminus = theta - epsilon
	J_plus = forward_prop(x, thetaplus)
	J_minus = forward_prop(x, thetaminus)
	grad_approx = (J_plus - J_minus) / (2 * epsilon)


def gradient_check_n(parameters, gradients, X, Y, epsilon=1e-7):

	parameters_v = dictionary_to_vector(parameters)
	grad_v = gradients_to_vector(gradients)
	num_parameters = parameters_v.shape[0]

	for i in range(num_parameters):
		thetaplus = np.copy(parameters_v)
		thetaplus[i][0] = thetaplus[i][0] + epsilon
		J_plus[i] = forward_prop(X, Y, vector_to_dictionary(thetaplus))


def load_dataset():
	np.random.seed(1)
	plt.scatter(train_X[:,0], train_X[:,1], c=train_Y, s=40, cmap=plt.cm.Spectral)

def plot_decision_boundary(model, X, Y):
	x_min, x_max = X[0,:].min() - 1, X[0,:].max() + 1
	y_min, y_max = X[1,:].min() - 1, X[1,:].max() + 1
	h = 0.01

	xx, yy = np.meshgrid(np.arange(x_min,x_max,h), np.arange(y_min, y_max,h))

	Z = model(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)

	plt.contourf(xx, yy, Z, cmap)
	plt.scatter(X[0,:], X[1,:], c=y, cmap)

data =scipy.io.loadmat('xxx.mat')

def dictionary_to_vectgor(parameters):

	keys = []
	for key in parameters.keys():

		new_vector = np.reshape(parameters[key], (-1,1))

		keys = keys + [key]*new_vector.shape[0]

		if count == 0:
			theta = new_vector
		else:
			theta = np.append(theta, new_vector, axis=0)
			theta = np.concatenate((theta, new_vector), axis=0)
		count += 1

def vector_to_dictionary(theta):
	parameters = {}
	parameters['W1'] = theta[:2].reshape()

%load_ext autoreload
%autoreload 2

y_0 = np.where(y==0, 1, 0)
y_0 = (y==0).astype(int)

i = np.where(p[0,:]==1)
