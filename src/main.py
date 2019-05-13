from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ShuffleSplit
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import adult_preprocessing as adult
import dutch_preprocessing as dutch
from exp_plot import exp2 as exp2_plot
from fair_classification.classifiers import *

logging.basicConfig (filename='main.log', level=logging.INFO)
logger = logging.getLogger (__name__)


def exp1():
	# corresponding to Section 4.2
	I = 3  # number of datasets
	J = 5  # number of splits
	K = 4  # number of classifiers

	risk_diff_result = np.zeros ((I, K + 2, J))

	def build_max_min_classifiers(s_var, y_var, x, s, y):
		optimal_clf = BayesUnfairClassifier (y_var, s_var, 'zero_one', 'zero_one')
		optimal_clf.fit (x, y, s)
		rd_min = optimal_clf.rd_min_kappa
		rd_max = optimal_clf.rd_max_delta
		risk_diff_result[i, 0:2, j] = rd_max, rd_min

	def build_test_classifier(s_var, y_var, x, s, y, clf, train_index, test_index):
		# build classifier on the training set
		clf.fit (x.ix[train_index,], y.ix[train_index,])

		# test the classifier on the testing set
		y_ = clf.predict (x.ix[test_index,])

		# validate risk and accuracy
		_, risk = compute_risk (y.ix[test_index,], y_, 'zero_one')
		accuracy = accuracy_score (y.ix[test_index,], y_)
		assert abs (risk + accuracy - 1) <= 0.001

		# compute risk difference
		_, rd = compute_risk_difference (s.ix[test_index,], y_, s_var, 'zero_one')
		risk_diff_result[i, k + 2, j] = rd

	for i, (s_var, y_var, x, s, y, _) in enumerate ([adult.load_data (), dutch.load_data (), adult.load_clean_data ()]):
		print ('-' * 40)
		print ('Statistics:')
		print ('# of instances: %d' % x.shape[0])
		print ('# of features: %d' % x.shape[1])
		print ('-' * 40)

		rs = ShuffleSplit (n_splits=5, test_size=0.2, random_state=54321)

		for j, (train_index, test_index) in enumerate (rs.split (x)):
			# build the Maximum/Minimum RD Classifiers on the original data
			build_max_min_classifiers (s_var, y_var, x.ix[train_index,], s.ix[train_index,], y.ix[train_index,])
			# build the logistic regression model on the original data
			for k, clf in enumerate ([LogisticRegression (random_state=rand), SVC (random_state=rand), DecisionTreeClassifier (random_state=rand), BernoulliNB ()]):
				build_test_classifier (s_var, y_var, x, s, y, clf, train_index, test_index)

	rd_mean = np.mean (risk_diff_result, axis=2)
	# rd_std = np.std (risk_diff_result, axis=2)
	np.set_printoptions (precision=3, suppress=True)
	print (rd_mean.T)
	np.savetxt ('outputs/bayes_unfair.txt', rd_mean.T, delimiter=',')


def train_measure(clf, x, y, s, s_var, rs):
	risk_list = []
	rd_list = []
	# cross validation and compute averages of risk and risk difference
	for train_index, test_index in rs.split (x):
		clf.fit (x.ix[train_index,], y.ix[train_index,], s.ix[train_index,])
		h_ = np.dot (x.ix[test_index,], clf.coef_) + clf.intercept_
		_, risk = compute_risk (y.ix[test_index,], h_, 'zero_one')
		_, rd = compute_risk_difference (s.ix[test_index,], h_, s_var, 'zero_one')
		risk_list.append (risk)
		rd_list.append (rd)
	return sum (risk_list) / len (risk_list), sum (rd_list) / len (rd_list)


def exp2():
	# corresponding to Section 4.3
	print ('%s %s %s' % ('=' * 10, 'begin', '=' * 10))
	for name, (s_var, y_var, x, s, y, offset) in [('adult', adult.load_data ()), ('dutch', dutch.load_data ())]:
		# five-fold cross-validation
		# dccp is too slow.
		rs = ShuffleSplit (n_splits=5, test_size=0.2, random_state=12345)
		for phi_name in ['logistic']:

			# unconstrained classifier
			clf = MarginClassifier (y_var, s_var, 'zero_one', 'zero_one', phi_name)
			risk, rd = train_measure (clf, x, y, s, s_var, rs)
			fout = open ('outputs/%s-unconstrained.txt' % name, 'w')
			fout.write ('%f, %f\n' % (risk, rd))
			fout.close ()

			# Zafar 2017a and Zafar 2017b
			for kappa_name in ['Zafar_identity', 'Zafar_hinge']:
				dict_identify = []
				for m in np.linspace (0.05, 1, 19, endpoint=False):
					logger.info ('Working on m=%0.3f' % m)
					clf = ZafarFairClassifier (y_var, s_var, kappa_name, kappa_name, phi_name, m=m)
					risk, rd = train_measure (clf, x, y, s, s_var, rs)
					dict_identify.append ([m, risk, rd])
				np.savetxt ('outputs/%s-%s.txt' % (name, kappa_name), np.array (dict_identify), delimiter=',')

			# Fairness-aware classification
			for kappa_name, delta_name in [('hinge', 'hinge')]:  # , ('squared', 'squared')]:
				dict_kappa_delta = []
				tau_name = '%s-%s-tau' % (kappa_name, delta_name)
				for tau in np.linspace (offset[tau_name][0], offset[tau_name][1], 50):
					clf = FairClassifier (y_var, s_var, kappa_name, delta_name, phi_name, tau1=tau, tau2=-tau - 20)
					risk, rd = train_measure (clf, x, y, s, s_var, rs)
					dict_kappa_delta.append ([tau, risk, rd])
				np.savetxt ('outputs/%s-%s.txt' % (name, kappa_name), np.array (dict_kappa_delta), delimiter=',')

	print ('%s %s %s' % ('=' * 10, 'end', '=' * 10))


if __name__ == '__main__':
	rand = np.random.RandomState (42)
	exp1 ()
	exp2 ()
	exp2_plot ()
