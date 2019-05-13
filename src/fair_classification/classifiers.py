from abc import ABC, abstractmethod

import dccp
import cvxpy as cvx
import pandas as pd
from sklearn.model_selection import KFold

from fair_classification.functions import *

logging.basicConfig (level=logging.INFO)
logger = logging.getLogger (__name__)


def get_key(name):
	if name.__len__ () > 1:
		return tuple (name)
	else:
		return name[0]


class AbstractClassifier (ABC):

	def __init__(self, y_var, s_var, kappa_name='zero_one', delta_name='zero_one'):
		super ().__init__ ()
		self.y_var = y_var
		self.s_var = s_var
		self.kappa_name = kappa_name
		self.delta_name = delta_name

	@abstractmethod
	def fit(self, x, y, s=None):
		pass

	@abstractmethod
	def predict(self, x):
		pass

	def cross_validation(self, x, y, s, fold=5, seed=0):
		kf = KFold (n_splits=fold, shuffle=True, random_state=seed)
		risk = 0.0
		rd = 0.0
		for train_index, test_index in kf.split (x):
			self.fit (x.ix[train_index, :], y.ix[train_index], s.ix[train_index])
			h_ = self.predict (x.ix[test_index, :])
			_, risk_ = compute_risk (y.ix[test_index], h_, 'zero_one')
			_, rd_ = compute_risk_difference (s.ix[test_index], h_, self.s_var, 'zero_one')
			risk += risk_
			rd += rd_
		return risk / fold, rd / fold


class BayesUnfairClassifier (AbstractClassifier):
	rd_min_kappa: float

	def __init__(self, y_var, s_var, kappa_name='zero_one', delta_name='zero_one', phi_name=None, ):
		super ().__init__ (y_var, s_var, kappa_name, delta_name)
		self.rules = {}
		self.x_name = []
		self.p = 0.0
		self.rd_min_kappa = 0.0
		self.rd_max_delta = 0.0

	def fit(self, x, y, s=None):
		self.x_name = sorted (x.columns)
		self.rules = {}
		df = pd.concat ([x, y, s], axis=1)
		total, pos_num, neg_num = self.s_var.count (df)
		self.p = pos_num / total

		for key, group in df.groupby (self.x_name):
			total, pos_num, neg_num = self.s_var.count (group)
			eta_s = pos_num / total

			crd, self.rules[key] = compute_minimal_crd (eta_s, self.p, self.kappa_name)
			self.rd_min_kappa += crd * total / df.__len__ ()

			crd, _ = compute_maximal_crd (eta_s, self.p, self.delta_name)
			self.rd_max_delta += crd * total / df.__len__ ()

	def predict(self, x):
		y_ = pd.Series (data=np.zeros (x.shape[0]), name='pred')
		x_ = x[self.x_name]
		for i in range (x_.shape[0]):
			key = get_key (x_.iloc[i])
			try:
				y_.iloc[i] = self.rules[key]
			except KeyError:
				logger.error ('%s is not found in the rules' % key)
				raise KeyError
		return y_


class MarginClassifier (AbstractClassifier):
	def __init__(self, y_var, s_var, kappa_name='zero_one', delta_name='zero_one', phi_name='logistic'):
		super ().__init__ (y_var, s_var, kappa_name, delta_name)
		self.phi_name = phi_name
		self.intercept_ = 0.0
		self.coef_ = np.empty (0)

	def switch(self):
		if self.phi_name == 'logistic':
			self.cvx_phi = lambda z: cvx.logistic (-z) / math.log (2.0, math.e)
		elif self.phi_name == 'hinge':
			self.cvx_phi = lambda z: cvx.pos (1.0 - z)
		elif self.phi_name == 'squared':
			self.cvx_phi = lambda z: cvx.square (-z)
		elif self.phi_name == 'exponential':
			self.cvx_phi = lambda z: cvx.exp (-z)
		else:
			logger.error ('%s is not include' % self.phi_name)
			logger.info ('Logistic is the default setting')
			self.cvx_phi = lambda z: cvx.logistic (-z) / math.log (2, math.e)

	def preprocess(self, x):
		self.x_ = np.c_[x, np.ones ([x.shape[0], 1])]
		self.w = cvx.Variable (self.x_.shape[1])

	def optimize(self):
		self.prob.solve (solver=cvx.ECOS, feastol=1e-8, abstol=1e-8, reltol=1e-8, max_iters=200, verbose=False, warm_start=False)
		print ('status %s ' % self.prob.status)
		if self.prob.status == cvx.OPTIMAL:
			self.coef_ = self.w.value.A1[:-1]
			self.intercept_ = self.w.value.A1[-1]
		else:
			logger.error ('Infeasible')
			raise TypeError

	def fit(self, x, y, s=None):
		self.switch ()
		self.preprocess (x)
		# construct objective function
		yz = cvx.mul_elemwise (y.values, self.x_ * self.w)
		objective = cvx.Minimize (cvx.sum_entries (self.cvx_phi (yz)))
		self.prob = cvx.Problem (objective)
		# solve the optimization problem
		self.optimize ()

	def predict(self, x):
		h = np.dot (x, self.coef_) + self.intercept_
		vfunc = np.vectorize (lambda i: self.y_var.pos if i >= 0 else self.y_var.neg)
		y_ = vfunc (h)
		return y_  # [:, 0]


class FairClassifier (MarginClassifier):

	def __init__(self, y_var, s_var, kappa_name='zero_one', delta_name='zero_one', phi_name='logistic', tau1=0.05, tau2=-0.05):
		super ().__init__ (y_var, s_var, kappa_name, delta_name, phi_name)
		self.tau1 = tau1
		self.tau2 = tau2
		self.p = 0.0
		self.intercept_ = 0.0
		self.coef_ = np.empty (0)

	def switch(self):
		if self.phi_name == 'logistic':
			self.cvx_phi = lambda z: cvx.logistic (-z)  # / math.log(2, math.e)
		elif self.phi_name == 'hinge':
			self.cvx_phi = lambda z: cvx.pos (1 - z)
		elif self.phi_name == 'squared':
			self.cvx_phi = lambda z: cvx.square (-z)
		elif self.phi_name == 'exponential':
			self.cvx_phi = lambda z: cvx.exp (-z)
		else:
			logger.error ('%s is not include' % self.phi_name)
			logger.info ('Logistic is the default setting')
			self.cvx_phi = lambda z: cvx.logistic (-z)  # / math.log(2, math.e)

		if self.kappa_name == 'logistic':
			self.cvx_kappa = lambda z: cvx.logistic (z)  # / math.log(2, math.e)
			self.psi_kappa = lambda mu: ((1 + mu) * math.log (1 + mu) + (1 - mu) * math.log (3 - mu)) / 2
		elif self.kappa_name == 'hinge':
			self.cvx_kappa = lambda z: cvx.pos (1 + z)
			self.psi_kappa = lambda mu: mu
		elif self.kappa_name == 'squared':
			self.cvx_kappa = lambda z: cvx.square (1 + z)
			self.psi_kappa = lambda mu: mu ** 2
		elif self.kappa_name == 'exponential':
			self.cvx_kappa = lambda z: cvx.exp (z)
			self.psi_kappa = lambda mu: 1 - math.sqrt (1 - mu ** 2)
		else:
			logger.error ('%s is not include' % self.kappa_name)
			logger.info ('hinge is the default setting')
			self.cvx_kappa = lambda z: cvx.pos (1 + z)
			self.psi_kappa = lambda mu: mu

		if self.delta_name == 'logistic':
			self.cvx_delta = lambda z: 1 - cvx.logistic (-z)  # / math.log(2, math.e)
			self.psi_delta = lambda mu: ((1 + mu) * math.log (1 + mu) + (1 - mu) * math.log (1 - mu)) / 2
		elif self.delta_name == 'hinge':
			self.cvx_delta = lambda z: 1 - cvx.pos (1 - z)
			self.psi_delta = lambda mu: mu
		elif self.delta_name == 'squared':
			self.cvx_delta = lambda z: 1 - cvx.square (1 - z)
			self.psi_delta = lambda mu: mu ** 2
		elif self.delta_name == 'exponential':
			self.cvx_delta = lambda z: 1 - cvx.exp (-z)
			self.psi_delta = lambda mu: 1 - math.sqrt (1 - mu ** 2)
		else:
			logger.error ('%s is not include' % self.delta_name)
			logger.info ('hinge is the default setting')
			self.cvx_delta = lambda z: cvx.pos (1 + z)
			self.psi_delta = lambda mu: mu

	def count(self, x, y, s):
		df = pd.concat ([x, y, s], axis=1)
		total, pos_num, neg_num = self.s_var.count (df)
		self.p = pos_num / total
		self.pos_num = pos_num
		self.neg_num = neg_num

	def fit(self, x, y, s=None):
		self.switch ()
		self.count (x, y, s)
		self.preprocess (x)
		# objective function
		yz = cvx.mul_elemwise (y.values, self.x_ * self.w)
		objective = cvx.Minimize (cvx.sum_entries (self.cvx_phi (yz)))
		# constraints
		weight = s.apply (lambda val: 1.0 / self.pos_num if val == self.s_var.pos else 1.0 / self.neg_num)
		sz = cvx.mul_elemwise (s.values, self.x_ * self.w)
		constraints = [
			(cvx.sum_entries (cvx.mul_elemwise (weight.values, self.cvx_kappa (sz)))) - 1 <= self.tau1,
			(cvx.sum_entries (cvx.mul_elemwise (weight.values, self.cvx_delta (sz)))) - 1 >= self.tau2
		]
		self.prob = cvx.Problem (objective, constraints)
		self.optimize ()


class ZafarFairClassifier (FairClassifier):
	def __init__(self, y_var, s_var, kappa_name='zero_one', delta_name='zero_one', phi_name='logistic', m=0.0, c=0.0):
		super ().__init__ (y_var, s_var, kappa_name, delta_name, phi_name)
		self.m = m
		self.c = c
		self.p = 0.0
		self.intercept_ = 0.0
		self.coef_ = np.empty (0)

	def switch(self):
		if self.phi_name == 'logistic':
			self.cvx_phi = lambda z: cvx.logistic (-z) / math.log (2, math.e)
		else:
			self.cvx_phi = lambda z: cvx.pos (1 - z)
		if self.kappa_name == 'Zafar_hinge':
			self.cvx_kappa = lambda z: cvx.pos (z)
		else:
			self.cvx_kappa = lambda z: z

	def unconstrained(self, x, y, s):
		clf = MarginClassifier (self.y_var, self.s_var, 'zero_one', 'zero_one', self.phi_name)
		clf.fit (x, y, self.phi_name)
		h_ = np.dot (x, clf.coef_) + clf.intercept_
		_, _, RD_kappa, _ = compute_risk_difference (s, h_, self.s_var, self.kappa_name, self.kappa_name)
		self.c = RD_kappa + 1

	def optimize(self):
		if self.kappa_name == 'Zafar_hinge':
			try:
				self.prob.solve (method='dccp')#, feastol=1e-4, abstol=1e-4, reltol=1e-4, max_iters=200, verbose=False, warm_start=False)
				logger.info ('DCCP')
			except cvx.error.SolverError:
				logger.error ('DCCP infeasible')
				raise cvx.SolverError
		else:
			self.prob.solve (solver=cvx.ECOS, feastol=1e-4, abstol=1e-4, reltol=1e-4, max_iters=200, verbose=False, warm_start=False)
			logger.info ('CVXPY')
		logger.info ('status %s ' % self.prob.status)

		if self.prob.status == cvx.OPTIMAL or self.prob.status == 'Converged':
			self.coef_ = self.w.value.A1[:-1]
			self.intercept_ = self.w.value.A1[-1]
		else:
			logger.error ('Infeasible')
			raise TypeError

	def fit(self, x, y, s=None):
		self.unconstrained (x, y, s)
		self.switch ()
		self.count (x, y, s)
		self.preprocess (x)
		# objective function
		yz = cvx.mul_elemwise (y.values, self.x_ * self.w)
		objective = cvx.Minimize (cvx.sum_entries (self.cvx_phi (yz)))
		# constraints
		weight = s.apply (lambda val: 1.0 / self.pos_num if val == self.s_var.pos else 1.0 / self.neg_num)
		sz = cvx.mul_elemwise (s.values, self.x_ * self.w)
		constraints = [
			(cvx.sum_entries (cvx.mul_elemwise (weight.values, self.cvx_kappa (sz)))) <= self.c * self.m,
			(cvx.sum_entries (cvx.mul_elemwise (weight.values, self.cvx_kappa (sz)))) >= -self.c * self.m - 20
		]

		self.prob = cvx.Problem (objective, constraints)
		self.optimize ()
