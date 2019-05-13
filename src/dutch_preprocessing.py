import os

from imblearn.over_sampling import RandomOverSampler

from fair_classification.attribute import *
from fair_classification.classifiers import *

logging.basicConfig (level=logging.INFO)
logger = logging.getLogger (__name__)


def load_data():
	src_path = os.path.dirname (os.path.realpath (__file__))
	df = pd.read_csv (os.path.join (src_path, '../data/dutch/dutch.csv'))

	s_var = BinaryVariable (name=u'sex', pos=2, neg=1)
	y_var = BinaryVariable (name=u'occupation', pos=u'5_4_9', neg=u'2_1')
	x_vars = [
		QuantitativeVariable ('age'),
		CategoricalVariable ('household_position'),
		CategoricalVariable ('household_size'),
		QuantitativeVariable ('prev_residence_place'),
		CategoricalVariable ('citizenship'),
		CategoricalVariable ('country_birth'),
		QuantitativeVariable ('edu_level'),
		CategoricalVariable ('economic_status'),
		CategoricalVariable ('cur_eco_activity'),
		CategoricalVariable ('Marital_status')
	]

	s = s_var.normalize (df[s_var.name])
	y = y_var.normalize (df[y_var.name])

	x = pd.DataFrame (data=None)
	for x_var in x_vars:
		x = pd.concat ([x, x_var.normalize (x=df[x_var.name])], axis=1)

	xs = pd.concat ([x, s], axis=1)
	ros = RandomOverSampler (random_state=0)
	X_resampled, y_resampled = ros.fit_sample (xs, y)
	x = pd.DataFrame (X_resampled[:, :-1])
	s = pd.Series (X_resampled[:, -1], name=s_var.name)
	y = pd.Series (y_resampled, name=y_var.name)

	offset = {
		'hinge-hinge-tau': [0.51, 2.54],
	}
	return s_var, y_var, x, s, y, offset
