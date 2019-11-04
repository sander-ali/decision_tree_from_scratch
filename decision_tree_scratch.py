# CART on the Bank Note dataset
from random import seed
from random import randrange
from csv import reader
import matplotlib.pyplot as plt
 
# Load a CSV file
def csv_read(filename):
	file_name = open(filename, "rt")
	read_line = reader(file_name)
	const_data = list(read_line)
	return const_data
 
# transforming string values to float
def str_to_float(const_data, col):
	for row in const_data:
		row[col] = float(row[col].strip())
 
# dividing data into k subsets
def split_cv(const_data, kfold):
	const_data_split = list()
	const_data_copy = list(const_data)
	k = int(len(const_data) / kfold)
	for i in range(kfold):
		num_fold = list()
		while len(num_fold) < k:
			index = randrange(len(const_data_copy))
			num_fold.append(const_data_copy.pop(index))
		const_data_split.append(num_fold)
	return const_data_split
 
# Calculate accuracy percentage
def acc_p(gt, pred):
	pos = 0
	for i in range(len(gt)):
		if gt[i] == pred[i]:
			pos += 1
	return pos / float(len(gt)) * 100.0
 
def alg_eval(const_data, method, kfold, *args):
	num_fold = split_cv(const_data, kfold)
	scores = list()
	for k in num_fold:
		data_train = list(num_fold)
		data_train.remove(k)
		data_train = sum(data_train, [])
		data_test = list()
		for row in k:
			row_copy = list(row)
			data_test.append(row_copy)
			row_copy[-1] = None
		pred = method(data_train, data_test, *args)
		gt = [row[-1] for row in k]
		acc = acc_p(gt, pred)
		scores.append(acc)
	return scores
 
# Split a dataset based on an attribute and an attribute value
def test_split(idx, val, const_data):
	left, right = list(), list()
	for row in const_data:
		if row[idx] < val:
			left.append(row)
		else:
			right.append(row)
	return left, right
# Calculate the Gini index for a split dataset
def idx_gini(groups, classes):
    instances = float(sum([len(group) for group in groups]))
    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        for val_class in classes:
            p = [row[-1] for row in group].count(val_class) / size
            score += p * p
        gini += (1.0 - score) * (size / instances)
    return gini

def get_split(const_data):
	val_classes = list(set(row[-1] for row in const_data))
	idx_b, val_b, score_b, groups_b = 999, 999, 999, None
	for idx in range(len(const_data[0])-1):
		for row in const_data:
			groups = test_split(idx, row[idx], const_data)
			gini = idx_gini(groups, val_classes)
			if gini < score_b:
				idx_b, val_b, score_b, groups_b = idx, row[idx], gini, groups
	return {'index':idx_b, 'value':val_b, 'groups':groups_b}
 
# Create a terminal node value
def terminal_node(group):
	out = [row[-1] for row in group]
	return max(set(out), key=out.count)
 
# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
	left, right = node['groups']
	del(node['groups'])
	# check for a no split
	if not left or not right:
		node['left'] = node['right'] = terminal_node(left + right)
		return
	# check for max depth
	if depth >= max_depth:
		node['left'], node['right'] = terminal_node(left), terminal_node(right)
		return
	# process left child
	if len(left) <= min_size:
		node['left'] = terminal_node(left)
	else:
		node['left'] = get_split(left)
		split(node['left'], max_depth, min_size, depth+1)
	# process right child
	if len(right) <= min_size:
		node['right'] = terminal_node(right)
	else:
		node['right'] = get_split(right)
		split(node['right'], max_depth, min_size, depth+1)
 
# Build a decision tree
def tree_develop(train, max_depth, min_size):
	root = get_split(train)
	split(root, max_depth, min_size, 1)
	return root
 
# Make a prediction with a decision tree
def prediction(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return prediction(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return prediction(node['right'], row)
		else:
			return node['right']
 
# Classification and Regression Tree Algorithm
def dt_method(train, test, max_depth, min_size):
	tree = tree_develop(train, max_depth, min_size)
	preds = list()
	for row in test:
		pred = prediction(tree, row)
		preds.append(pred)
	return(preds)
 
# Test CART on Bank Note dataset
seed(1)
# load and prepare data
filename = 'data_banknote_authentication.csv'
const_data = csv_read(filename)
# convert string attributes to integers
for i in range(len(const_data[0])):
	str_to_float(const_data, i)
# evaluate algorithm
kfold = 10
max_depth = 8
min_size = 5
scores = alg_eval(const_data, dt_method, kfold, max_depth, min_size)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))