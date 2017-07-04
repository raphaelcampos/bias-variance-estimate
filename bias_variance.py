import numpy as np
from scipy.stats import mode

from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline

from sklearn.model_selection import StratifiedKFold, KFold

def rssCV(estimator, X, y, num_classifications, train_size, gamma):
    
    classes = np.unique(y)
    n_classes = classes.shape[0]
    
    if 0 >= train_size or train_size >= X.shape[0]:
        raise ValueError("Expected 0 < train_size < %d, given %d" % 
        																				(X.shape[0], train_size))

    if train_size/(X.shape[0] + 1.01) >= gamma or gamma >= 1.:
        raise ValueError("Expected %f <= gamma < 1, given %f" % 
        											(train_size/(X.shape[0] + 1), gamma))
    
    train_size = int(train_size)
    
    tps = np.ceil(train_size / gamma + 1.).astype(int)
    k = np.ceil(tps / (tps - train_size)).astype(int)
    q = np.floor(X.shape[0] / tps).astype(int)
    np.random.seed(0)
    index = np.arange(X.shape[0])
    np.random.shuffle(index)
    
    seg_size = index.shape[0] / q
    E = [index[(seg_size * (i)):(seg_size * (i + 1))] for i in range(q)]
    
    Eq1 = None
    if seg_size * q < index.shape[0]:
        Eq1 = index[(seg_size * q):]

    P = np.zeros((X.shape[0], n_classes))
    for l in range(num_classifications):
        for i, Ei in enumerate(E):
            kf = KFold(n_splits=k, shuffle=True, random_state = None)
            j = 0
            for train_index, test_index in kf.split(Ei):
                train_index = train_index[:train_size]
                
                X_train, y_train = X[Ei[train_index]], y[Ei[train_index]]
                X_test, y_test = X[Ei[test_index]], y[Ei[test_index]]
                
                estimator.fit(X_train, y_train)
                
                y_pred = estimator.predict(X_test).astype(int)
                P[Ei[test_index], y_pred] += 1
                
                if i == 0 and j == 0 and (not Eq1 is None):
                    X_test, y_test = X[Eq1], y[Eq1]
                    y_pred = estimator.predict(X_test)
                    P[Ei[test_index], y_pred] += 1
                j += 1
                
    return P

def bootstrap(estimator, X, y, M = 50, n_folds = 5):
	n_samples, n_features = X.shape

	classes = np.unique(y)
	n_classes = classes.shape[0]

	indices = np.arange(n_samples)

	times = np.zeros((n_samples))
	P = np.zeros((n_samples, n_classes))
	for m in range(int(M)):
		index = np.random.choice(indices,
							 size=n_samples, replace=True)
		
		sampling_freq = np.bincount(index)
		times[:sampling_freq.shape[0]] += sampling_freq 

		kf = KFold(n_splits=n_folds, 
					shuffle=True, random_state = None)
		 
		for train_index, test_index in kf.split(index, y[index]):
			X_train, y_train = X[index[train_index]], y[index[train_index]]
			X_test, y_test = X[index[test_index]], y[index[test_index]]

			y_proba = estimator.\
								fit(X_train, y_train).\
								predict_proba(X_test)

			for i, j in enumerate(index[test_index]):
				P[j, :] += y_proba[i]


	return P / times[:, np.newaxis]

def compute_statistics(P, X, y):
    classes = np.unique(y)
    n_classes = classes.shape[0]

    indices = np.arange(P.shape[0])

    # estimate bayes error
    knn = KNeighborsClassifier(n_neighbors = 3, metric='cosine',
    				 algorithm="brute", weights="uniform", n_jobs = -1)
    tfidf = TfidfTransformer(norm="max", sublinear_tf=False)
    
    knn = make_pipeline(StandardScaler(), knn).fit(X, y)
    
    P_target = np.zeros((X.shape[0], n_classes))
    kf = KFold(n_splits=10, 
    			shuffle=True, random_state = None)
    y_pred = np.zeros((X.shape[0]))
    for train_index, test_index in kf.split(X, y):
      P_target[test_index] = knn.predict_proba(X[test_index])
      # neighbors = knn.kneighbors(n_neighbors = 3,
      # 														return_distance=False)

      # y_pred, _ = mode(y[neighbors])

    # P_target = ((P_target * 4) - (classes == y[:, np.newaxis]).astype(int)) / 3.

    P_classifier = P / np.sum(P, axis = 1)[:,np.newaxis]
    
    SC = np.argmax(P_classifier, axis=1)
    ST = np.argmax(P_target, axis=1)
    
    Var_target =  (1 - P_target[indices, ST]).mean()
    Var_classifier = (1 - P_classifier[indices, SC]).mean()
    bias_classifier = (SC != ST).mean()
    ptpc = np.sum(np.multiply(P_target, P_classifier), axis = 1)
    SE = (P_target[indices, ST] - P_target[indices, SC]).mean()
#     SE = (1. - ptpc).mean()
    acc = (SC == y).mean()
    # VE = (1 - acc) - Var_target - SE
    VE = (P_target[indices, SC] - ptpc).mean()
 
    # SE = (1 - acc) - Var_target - VE
    return {"BIAS": bias_classifier * 100,
    				"VAR": Var_classifier * 100,
    				"SE": SE * 100, 
    				"VE": VE * 100,
    				"BE": Var_target * 100}
    
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.datasets import load_breast_cancer
import pandas as pd
def exp_james():
	"""
	Trying to reproduce James's paper experiments
	[ref] James, G. M. (2003). Variance and bias for general loss functions.
	 			Mach. Learn., 51(2):115--135.
	"""
	def load_dset(data, delimiter=","):
		"""
		helper for loading the datasets
		"""
		D = np.loadtxt(data, delimiter=delimiter)

		X, y = D[:, :-1], D[:,-1]

		enc = LabelEncoder()
		y = enc.fit_transform(y)

		return X, y

	def run(clfs, X, y):
		"""
		runs the experiments given the dataset and the classifiers
		"""
		results = {}
		for clf in clfs:
			label, estimator, params = clf
			for param, values in params.items():
				for value in values:
					estimator.set_params(**{param: value})
					train_size = np.floor(X.shape[0] * 0.5)
					P = rssCV(estimator, X, y, 50, train_size, 0.6)
					stat = compute_statistics(P, X, y.astype(int))
					results[label + str(value)] = stat
		
		df = pd.DataFrame(data=results)
		print(df.loc[["BIAS","VAR","SE", "VE", "BE"]])



	dt = DecisionTreeClassifier(max_leaf_nodes = 5)
	knn = KNeighborsClassifier(n_neighbors = 1, metric='euclidean',
    				 algorithm="kd_tree", weights="uniform", n_jobs = -1)
	
	knn = make_pipeline(StandardScaler(), knn)

	clfs = [
		("dt", dt, {'max_leaf_nodes': [5, 10]}),
		("knn", knn, {'kneighborsclassifier__n_neighbors': [1, 5, 11]}),
		]

	datas = [
		load_dset("datasets/glass.data.txt"),
		load_breast_cancer(True),
		load_dset("datasets/vowel-context.data.txt", " "),
		load_dset("datasets/dermatology.data.txt", ","),
		]

	for data in datas:
		X, y = data
		print("----------------------\n\n")
		run(clfs, X, y)
		print("----------------------\n\n")


if __name__ == '__main__':
	exp_james()
	exit()
	from sklearn.datasets import load_svmlight_file
	from sklearn.tree import DecisionTreeClassifier

	from IPython import embed

	dt = DecisionTreeClassifier(max_leaf_nodes = 5)
	knn = KNeighborsClassifier(n_neighbors = 1, metric='euclidean',
    				 algorithm="kd_treee", weights="uniform", n_jobs = -1)


	from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

	X, y = load_svmlight_file("../release/datasets/4uni.svm")

	D = np.loadtxt("../release/datasets/glass.data.txt", delimiter=",")
	X, y = D[:, 1:-1], D[:,-1] - 1

	from sklearn.preprocessing import LabelEncoder
	enc = LabelEncoder()
	y = enc.fit_transform(y)

	train_size = np.floor(X.shape[0] * 0.5)
	print(train_size / (X.shape[0] + 1), train_size)
	P = rssCV(dt, X, y, 10, train_size, 0.6)


	compute_statistics(P, X, y.astype(int))
	P_classifier = P / np.sum(P, axis = 1)[:,np.newaxis]
	SC = np.argmax(P_classifier, axis=1)
	print(SC == y).mean()
	
	embed()
