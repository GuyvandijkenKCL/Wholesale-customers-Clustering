from pandas import DataFrame, Series, read_csv
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score 
import matplotlib.pyplot as plt



def read_csv_2(data_file):
	dataFrame = read_csv(data_file, encoding='unicode_escape')
	dataFrame.drop(['Channel', 'Region'], axis=1, inplace=True)
	return dataFrame

def summary_statistics(df):
	return df.describe()


def standardize(df):
	# describe = df.describe()
	for column in df.columns:
		mean = df[column].mean()
		std = df[column].std()
		df[column] = df[column].map(lambda x: (x - mean) / std)
	return df

def kmeans(df, k):
	kmeans = KMeans(n_clusters = k, random_state = 1, n_init=1)
	kmeans.fit(df)
	y = Series(kmeans.fit_predict(df))
	return y


def agglomerative(df, k):
	clustering = AgglomerativeClustering(k)
	y = Series(clustering.fit_predict(df))
	return y

def clustering_score(X,y):
	return silhouette_score(X, y)


def cluster_evaluation(df):
	evaluation_data = {
		'Algorithm': [],
		'data': [],
		'k': [],
		'Silhouette Score': [],
	}

	algorithms = {
		'Kmeans':kmeans,
		'Agglomerative':agglomerative
	}
	datas = {
		"Original": df,
		"Standardized": standardize(df)
	}

	for data in ['Original', "Standardized"]:
		for algorithm in ['Kmeans', 'Agglomerative']:
			for k in [3, 5, 10]:
				y = algorithms[algorithm](datas[data], k)
				score = clustering_score(datas[data], y)
				evaluation_data['Algorithm'].append(algorithm)
				evaluation_data['data'].append(data)
				evaluation_data['k'].append(k)
				evaluation_data['Silhouette Score'].append(score)


	evaluationDataFrame = DataFrame(evaluation_data)
	return evaluationDataFrame

def best_clustering_score(rdf):
	return rdf['Silhouette Score'].max()

def scatter_plots(df):
	count = 0
	df = standardize(df)
	clusters = kmeans(df, k=3)
	count = 0
	list(plt.colormaps)
	# for column in df.columns:
	# 	for column2 in df.columns:
	for i in range(len(df.columns)):
		for j in range(i + 1, len(df.columns)):
			plt.figure()
			plt.scatter(df[df.columns[i]], df[df.columns[j]], c=clusters, cmap=list(plt.colormaps)[count])
			plt.savefig(f'plot{count}.png')
			count += 1