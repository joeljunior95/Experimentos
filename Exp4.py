import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from timeit import default_timer as timer
from threading import Thread
import pickle
from os.path import isfile

"""
Distance matrix
"""
def matrix_dist(Cols, metric, df, n_threads):
        dist = dict()
        X = Cols.copy()
        threads = list()
        factor = int(len(X)/n_threads)

        for n in range(n_threads):
                X_ = []
                if n < n_threads - 1:
                        X_ = X[:factor]
                        X = X[factor:]
                else:
                        X_ = X
                t = Thread(target=thread,args=(dist, X_, X, metric, df))
                t.start()
                threads.append(t)

        for t in threads:
                t.join()

        return dist

def save_matrix(name, obj):
	with open(name + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
	return
		
def load_matrix(name):
	with open(name + '.pkl', 'rb') as f:
		return pickle.load(f)
	
def thread(dist, part, X, metric, df):
	for x in part:
		dist[x] = dict()
		for y in X:
			dist[x][y] = metric(x,y,df)
	return

"""
Distance Measure
"""
def I(Ai,Aj,db):
    inf = 0
    for k in db[Ai].unique():
        for l in db[Aj].unique():
            Pr = len(db[(db[Ai] == k) & (db[Aj] == l)])/len(db[[Ai,Aj]])
            PrAi = len(db[db[Ai] == k])/len(db[Ai])
            PrAj = len(db[db[Aj] == l])/len(db[Aj])
            inf += Pr*np.log(Pr/(PrAi*PrAj)) if Pr != 0 else 0
    return inf

def H(Ai,Aj,db):
    ent = 0
    for k in db[Ai].unique():
        for l in db[Aj].unique():
            Pr = len(db[(db[Ai] == k) & (db[Aj] == l)])/len(db[[Ai,Aj]])
            ent += Pr*np.log(Pr) if Pr != 0 else 0
    return -ent

def R(Ai,Aj,db):
    return I(Ai,Aj,db)/H(Ai,Aj,db)

def MR(Ai, p = list(),db = pd.DataFrame()):
    mr = 0
    for Aj in p:
        mr += R(Ai,Aj,db)
    return mr

def MR_N(Ai, p = list(), M = dict()):
    mr = 0
    for Aj in p:
        mr += M[Ai][Aj]
    return mr

"""
Feature selection algorithm
"""
def ACA(k = 2, df = pd.DataFrame(), M = dict(), n = 5):
	feats = df.columns
	Modes = np.random.choice(feats,k,replace=False)
	for i in range(n):
		Clusters = [[m] for m in Modes]
		for f in feats:
			if f not in Modes:
				Rf = 0
				flag = None
				for m in range(len(Modes)):
					mode = Modes[m]
					Rfc = M[f][mode]
					if Rfc > Rf:
						Rf = Rfc
						flag = m
				if flag:
					Clusters[flag].append(f)

		nModes = []
		for C in Clusters:
			f = None
			Mm = 0
			for A in C:
				Ma = MR_N(A,C,M)
				if Ma > Mm:
					f = A
					Mm = Ma
			if f:
				nModes.append(f)
		if np.array_equal(nModes,Modes):
			break
		Modes = nModes

	return Modes, Clusters

"""
"""
def manager(df,M):
	ncluster = 2
	k_chosen=None
	features=None
	silh = 0

	for k in range(2, len(df.columns)):
		print("K =",k)
		s = timer()
		aca = ACA(k,df,M)
		e = timer()
		print("Elapsed time after running ACA:", e - s)
		X=df[aca[0]].copy()
		kmeans_model = KMeans(n_clusters=ncluster, random_state=1).fit(X)
		labels = kmeans_model.labels_
		mysilh = silhouette_score(X, labels, metric='euclidean')
		print("Silhouette Score:", mysilh)
		if mysilh > silh:
			silh = mysilh
			k_chosen = k
			features=aca[0]
		print('\n')
		
	print("K chosen =",k_chosen)
	print("Silhoutte Score chosen =", silh)
	print("Features chosen =",features)

	chosenDF = pd.DataFrame()
	if features is not None:
		chosenDF = df[features]
		
	return chosenDF

def main(base_dir, distance_matrix_name = 'matrixACA', distance_matrix_extension = '.pkl',n_threads = 1):
	distance_matrix_name = "../Matrizes_de_Distancia/" + distance_matrix_name
	base = pd.read_csv(base_dir, sep="\t")
	matrix = None
	if isfile(distance_matrix_name+distance_matrix_extension):
		print("[LOG] Loading distance matrix")
		matrix = load_matrix(distance_matrix_name)
	else:
		print("[LOG] Not able to load distance matrix.\n[LOG] Calculating matrix.")
		matrix = matrix_dist(base.columns, R, df,n_threads)
		save_matrix(distance_matrix_name, matrix)
	
	newDF = manager(base,matrix)
	newDF.to_csv("../Bases_geradas/Exp4"+base_dir,sep="\t")

def teste():
	print("it works!")
	return

if __name__ == '__main__':
	main(distance_matrix_name = 'matrixACA', base_dir = 'complete_processed.csv')
