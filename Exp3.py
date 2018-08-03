import pandas as pd
import numpy as np
import math
from scipy.spatial.distance import cosine
from scipy.stats import entropy 
from sklearn import metrics
from sklearn.cluster import KMeans
from timeit import default_timer as timer
from threading import Thread
import pickle
from os.path import isfile
import operator
from multiprocessing.pool import Pool
from multiprocessing import cpu_count
import operator

"""
Distance Measure
"""
def S_OLD(vi,vj,bag):
	global alpha
	norm = np.sqrt(sum([(bag[c].max()-bag[c].min())**2 for c in bag.columns]))
	Dij = euclidean(vi,vj) / norm
	if alpha is None:
		D_ = 0
		for i in range(0,len(bag.index)):
			for j in range(i+1, len(bag.index)):
				D_ += euclidean(bag.loc[i,:],bag.loc[j,:]) / norm

		D_ /= (len(bag.index) * (len(bag.index) - 1) / 2)
		
		alpha = -1 * np.log(0.5) / D_

	return np.e**(-1*alpha*Dij)

def S(vi,vj,bag):
	return np.sum([1/len(bag.columns) if vi[c] == vj[c] else 0 for c in bag.columns])

def Ent(D):
	E = 0.0
	inst = D.index
	for i in inst:
		vi = D.loc[i,:].values
		for j in inst:	
			vj = D.loc[j,:].values
			Sij = np.sum([1/len(D.columns) if vi[c] == vj[c] else 0 for c in range(len(vi))])
			Eij = (Sij*np.log2(Sij) + (1 - Sij)*np.log2(1 - Sij)) if Sij > 0 and Sij < 1 else 0
			E += Eij
	return -E
	
"""
Feature selection algorithm
"""
def SUD_OLD(D):
	T = D.columns.copy()
	features = []
	while len(T) > 0:
		print("[SUD LOG] Length o T =",len(T))
		small_E = None
		vk = None
		for v in T:
			Tv = [i for i in T if i != v]
			Etv = Ent(D[Tv])
			if small_E is None or Etv < small_E:
				small_E = Etv 
				vk = v
		features.insert(0,vk)
		T = [i for i in T if i != vk]
	return features

def SUD(D):
	T = D.columns.copy()
	features = []
	it = 0
	for v in T:
		start = timer()
		Tv = [i for i in T if i != v]
		Etv = Ent(D[Tv])	
		features.append((v,Etv))
		end = timer()
		print("[SUD LOG] Iteration",it,"Time elapsed:",end - start)
		it+=1
	return [i[0] for i in sorted(features, key=operator.itemgetter(1), reverse=True)]

def SUD_Paral(D):
	T = D.columns.copy()
	features = []
	threads = []
	n_count = cpu_count()
	factor = int(len(T)/n_count)
	it = 0 #iteration counter
	tc = 0 #thread counter
	"""
	for n in range(n_count):
		T_ = []
		if n < n_count - 1:
			T_ = T[:factor]
			T = T[factor:]
		else:
			T_ = T
		t = Thread(target=thread_SUD,args=(T_,D,features))
		t.start()
		threads.append(t)
	"""	
	for v in T:
		t = Thread(target=thread_SUD2,args=(v,D,features))
		print("[SUD LOG] Starting thread",tc)
		t.start()
		threads.append(t)
		tc+=1
	
	for t in threads:
		print("[SUD LOG] Iteration",it)
		start = timer()
		t.join()
		end = timer()
		print("Time elapsed:",end - start)
		it+=1
	
	return [i[0] for i in sorted(features, key=operator.itemgetter(1), reverse=True)]

def thread_SUD2(v, D, features):
	Tv = [i for i in D.columns if i != v]
	Etv = Ent(D[Tv])
	features.append((v,Etv))
	return
	
def thread_SUD(T, D, features):
	for v in T:
		Tv = [i for i in D.columns if i != v]
		Etv = Ent(D[Tv])
		features.append((v,Etv))
	return
	
"""
"""
def manager(df):
	print("Starting SUD")
	s = timer()
	featuresSUD = SUD_Paral(df)
	e = timer()
	print("Elapsed time after running SUD:", e - s)
	print("Ordem SUD:",featuresSUD)
	
	k_chosen=None
	features=None
	silh = 0
	ncluster = 2

	for k in range(ncluster,len(featuresSUD)+1):
		#print("K =",k)
		Result = featuresSUD[:k]
		X=df[Result]
		kmeans_model = KMeans(n_clusters=ncluster, random_state=1).fit(X)
		labels = kmeans_model.labels_
		mysilh = metrics.silhouette_score(X, labels, metric='euclidean')
		#print("Silhouette Score:", mysilh)

		if mysilh > silh:
			silh = mysilh
			k_chosen = k
			features=Result
		print("\n")
		
	print("K chosen =",k_chosen)
	print("Silhoutte Score chosen =", silh)
	print("Features chosen =",features)
	return df[features]
	
	
def main(base_dir):
	base = pd.read_csv(base_dir, sep="\t")
	newDF = manager(base)
	newDF.to_csv("../Bases_geradas/Exp3_main_processed.csv",sep="\t")
	return

def main2(base_dir):
	base = pd.read_csv(base_dir, sep="\t")
	
	t1 = timer()
	feats = SUD_Paral(base)
	t2 = timer()
	print("[SUD_Paral] Time spent =",t2-t1,"\n","Result feats("+str(len(feats))+")\n",feats)
	
	"""
	t1 = timer()
	ent = Ent(base)
	t2 = timer()
	print("Entropy:", ent, "\nTime elapsed:",t2-t1)
	"""
	return

if __name__ == '__main__':
	main2("main_processed.csv")
