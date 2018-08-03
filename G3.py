import Exp4 #ACA
import Exp3 #SUD
import Exp2 #Mitra

def main(op):
	base_dir="../Bases_geradas/complete_processed.csv"
	Exp = None
	if op == 2:
		Exp = Exp2
	elif op == 3:
		Exp = Exp3
	elif op == 4:
		Exp = Exp4
	else:
		return
	print("[LOG] Executing experiment",op)
	Exp.main(base_dir)
	
if __name__ == '__main__':
	main(3)
