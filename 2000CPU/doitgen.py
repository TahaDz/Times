import os
import FeaturesExtractor.myscript as feature





if __name__ == '__main__':
	
	#a = [32, 64,128, 256, 512]
	a = list(range(32,513,32))
	i = 0
	n_data = 2000
	for nq in a:
		for nr in a:
			for nq in a:
					if i < n_data:
						os.system("gcc Cfiles/doitgen.c -lOpenCL -lm -o Cfiles/doitgen -w -fcompare-debug-second")
						os.system("./Cfiles/doitgen 1 0 "+ str(nq)+" "+ str(nr)+" "+ str(nq))

						i += 1

	print(i)










