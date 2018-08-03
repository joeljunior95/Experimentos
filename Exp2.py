import pandas as pd
import numpy as np
import math
from scipy.spatial.distance import cosine
from sklearn import metrics
from sklearn.cluster import KMeans
from timeit import default_timer as timer
from threading import Thread
import pickle
from os.path import isfile
import operator

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
Distance Measure-lambda do artigo
"""
def mici(x,y,bag):
    varx = bag[x].var()#variancia de x escalar
    vary = bag[y].var()#variancie de y escalar
    return varx + vary - np.sqrt((varx + vary)**2 - 4*varx*vary*(1 - bag[x].corr(bag[y])**2))

"""
Feature selection algorithm
"""	
def mitraOLD(bag, k = None, M = dict()):
    #print("\n[Inicio do Algoritmo]")
    O = bag.columns

    #Step 1
    R = O.copy()
    k = k if k else len(O) - 1
    
    #Step 2
    small_rik = 10
    while k > 1:
        #print("init k = ", k)
        #print("R len =", len(R))
        F = None
        discard = None
        for i in range(len(R)):
            Fi = R[i]
            neighbors = []
            for j in range(0,len(R)):
                if j != i:
                    Fj = R[j]
                    neighbors.append([M[Fi][Fj],j]) 
            neighbors = sorted(neighbors,reverse=False)[:k]
            if neighbors[-1][0] < small_rik:
                F = i
                discard = [n[-1] for n in neighbors]

        #Step 3
        R = [R[i] for i in range(0,len(R)) if i not in discard] if discard else R
        epsilon = small_rik

        #Step 4
        k = (len(R) - 1) if k > (len(R) - 1) else k
        #print("mid k = ", k)
        #print("R len =", len(R),"\n")
       
        #Step 5
        if k <= 1:
            return R

        #Step 6
        while small_rik >= epsilon:
            k -= 1

            for i in range(0,len(R)):
                Fi = R[i]
                neighbors = []
                for j in range(0,len(R)):
                    if j != i:
                        Fj = R[j]
                        neighbors.append([M[Fi][Fj],j]) 
                neighbors = sorted(neighbors,reverse=False)[:k]
                if neighbors[-1][0] < small_rik:
                    small_rik = neighbors[-1][0]
                    break

            if k <= 1:
                return R
    
        #Step 7
        #return to step 2
    
    #Step 8
    return R

def mitra(bag, k = None, M = dict(), n = 100):
	features = bag.columns.copy()
	error = None
	for iteration in range(n):
		feat_chosen = None
		cluster_chosen = None
		dist_chosen = None
		decr = 0
		for x in features:
			neighbors = [i for i in sorted(M[x].items(), key=operator.itemgetter(1)) if i[0] != x and i[0] in features]
			cluster = neighbors[:k]
			if len(cluster) == len(features) - 1:
				return features
			dist = cluster[-1][1]
			if dist_chosen is None or dist < dist_chosen:
				dist_chosen = dist
				cluster_chosen = [c[0] for c in cluster]
				feat_chosen = x
			if iteration > 0 and dist > error:
				decr -= 1
		
		features = [f for f in features if f not in cluster_chosen]
	
		if iteration == 0:
			error = dist_chosen
		else:
			k += decr
		
		if k < 1:
			return features
	
	return features

def manager(df,M):
	ncluster = 2
	k_chosen=None
	features=None
	silh = 0
	for k in range(2,len(df.columns)):
		print("K =",k)
		start = timer()
		Result = mitra(df,k,M)
		end = timer()
		print("Elapsed time:", end - start)
		print("Number of features",len(Result))
		X=df[Result]
		kmeans_model = KMeans(n_clusters=ncluster, random_state=1, n_jobs=1).fit(X)#agrupamemto kmeans
		labels = kmeans_model.labels_
		mysilh = metrics.silhouette_score(X, labels, metric='euclidean')#metrics euclidian
		print("Silhouette Score:", mysilh)
		if mysilh > silh:
			silh = mysilh
			k_chosen = k
			features=Result
		print("\n")
		
	print("K chosen =",k_chosen)
	print("Silhoutte Score chosen =", silh)
	print("Features chosen =",features)

	return df[features]
	
def main(base_dir, distance_matrix_name = 'matrixMITRA', distance_matrix_extension = '.pkl', n_threads = 1):
	distance_matrix_name = "../Matrizes_de_Distancia/" + distance_matrix_name
	base = pd.read_csv(base_dir, sep="\t")
	matrix = None
	if isfile(distance_matrix_name+distance_matrix_extension):
		print("[LOG] Loading distance matrix")
		matrix = load_matrix(distance_matrix_name)
	else:
		print("[LOG] Not able to load distance matrix.\n[LOG] Calculating matrix.")#Calculating matrix
		matrix = matrix_dist(base.columns, mici, base, n_threads)
		save_matrix(distance_matrix_name, matrix)
	
	newDF = manager(base,matrix)
	newDF.to_csv("../Bases_geradas/Exp2_main_processed.csv",sep="\t")

def teste(base_dir):
	base = pd.read_csv(base_dir, sep="\t")
	print("Calculating how much time does it take to measure distance with MICI:")
	n = 500
	print("Using",n,"features")
	df = base[base.columns[:n]]
	s = timer()
	for x in df.columns:
		for y in df.columns:
			mici(x,y,df)
	e = timer()
	print("Result:",e-s)

if __name__ == '__main__':
	main("complete_processed.csv")
