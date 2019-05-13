import os

from imblearn.over_sampling import RandomOverSampler

from fair_classification.attribute import *
from fair_classification.classifiers import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def mapping(tuple):
	# age, 37
	tuple['age'] = 1 if tuple['age'] > 37 else 0
	# workclass
	tuple['workclass'] = 'NonPrivate' if tuple['workclass'] != 'Private' else 'Private'
	# edunum
	tuple['education-num'] = 1 if tuple['education-num'] > 9 else 0
	# maritial statue
	tuple['marital-status'] = "Marriedcivspouse" if tuple[
														'marital-status'] == "Married-civ-spouse" else "nonMarriedcivspouse"
	# occupation
	tuple['occupation'] = "Craftrepair" if tuple['occupation'] == "Craft-repair" else "NonCraftrepair"
	# relationship
	tuple['relationship'] = "NotInFamily" if tuple['relationship'] == "Not-in-family" else "InFamily"
	# race
	tuple['race'] = 'NonWhite' if tuple['race'] != "White" else 'While'
	# hours per week
	tuple['hours-per-week'] = 1 if tuple['hours-per-week'] > 40 else 0
	# native country
	tuple['native-country'] = "US" if tuple['native-country'] == "United-States" else "NonUS"
	return tuple


def binarize():
	src_path = os.path.dirname(os.path.realpath(__file__))
	df = pd.read_csv(os.path.join(src_path, '../data/adult.csv'))
	df = df.drop(['fnlwgt', 'education', 'capital-gain', 'capital-loss'], axis=1)
	df = df.apply(mapping, axis=1)
	df.to_csv(os.path.join(src_path, '../data/adult-b.csv'), index=False)


def load_data(binary=False):
	src_path = os.path.dirname(os.path.realpath(__file__))

	s_var = BinaryVariable(name=u'sex', pos=u'Male', neg=u'Female')
	y_var = BinaryVariable(name=u'income', pos=u'>50K', neg=u'<=50K')

	if binary:
		df = pd.read_csv(os.path.join(src_path, '../data/adult/adult-b.csv'))
		x_vars = [
			CategoricalVariable('age'),
			CategoricalVariable('workclass'),
			CategoricalVariable('education-num'),
			CategoricalVariable('marital-status'),
			CategoricalVariable('occupation'),
			CategoricalVariable('relationship'),
			CategoricalVariable('race'),
			CategoricalVariable('hours-per-week'),
			CategoricalVariable('native-country')
		]
	else:
		df = pd.read_csv(os.path.join(src_path, '../data/adult/adult.csv'))
		x_vars = [
			QuantitativeVariable('age'),
			CategoricalVariable('workclass'),
			QuantitativeVariable('education-num'),
			CategoricalVariable('marital-status'),
			CategoricalVariable('occupation'),
			CategoricalVariable('relationship'),
			CategoricalVariable('race'),
			QuantitativeVariable('hours-per-week'),
			CategoricalVariable('native-country')
		]
	s = s_var.normalize(df[s_var.name])
	y = y_var.normalize(df[y_var.name])

	x = pd.DataFrame(data=None)
	for x_var in x_vars:
		x = pd.concat([x, x_var.normalize(x=df[x_var.name])], axis=1)

	xs = pd.concat([x, s], axis=1)
	ros = RandomOverSampler(random_state=0)
	X_resampled, y_resampled = ros.fit_sample(xs, y)
	x = pd.DataFrame(X_resampled[:, :-1])
	s = pd.Series(X_resampled[:, -1], name=s_var.name)
	y = pd.Series(y_resampled, name=y_var.name)

	offset = {
		'hinge-hinge-tau': [0.13, 3.27],
	}
	return s_var, y_var, x, s, y, offset


def load_clean_data():
	s_var, y_var, x, s, y, offset = load_data(True)
	s = s.sample(frac=1.0, random_state=2018)
	s.index = range(s.__len__())
	return s_var, y_var, x, s, y, offset