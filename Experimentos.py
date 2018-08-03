import sys, getopt
import fsa 
def main(argv):
	mtx = mtxdir = ''
	base = basedir = ''
	fsa = None
	try:
		opts, args = getopt.getopt(argv,"hb:a:m:",["basedir=","mtxdir="])
	except getopt.GetoptError:
		print('Experimentos.py -b <base> -a <algorithm>')
		sys.exit(2)
	
	for opt, arg in opts:
		if opt == '-h':
			print('Experimentos.py -b <base> -a <algorithm>')
		elif opt == "-a":
			fsa = fsa[arg.lower()]
		elif opt == "-b":
			arg = arg.lower()
			if arg in ('g1','main_processed'):
				base = 'main_processed.csv'
			elif arg in ('g2','desc'):
				base = 'desc.csv'
			elif arg in ('g3','complete_processed'):
				base = 'complete_processed.csv'
		elif opt == "-m":
			mtx = arg
		elif opt == "--basedir":
			basedir = arg
		elif opt == "--mtxdir":
			mtxdir = arg
	
	fsa #continuar daqui...
	"""
	criar a library/package fsa com todos os algoritmos dentro
	transformar cada algoritmo em uma classe diferente
	"""
			
	print(opts)
if __name__ == '__main__':
	main(sys.argv[1:])