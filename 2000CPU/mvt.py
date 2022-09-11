import os
import FeaturesExtractor.myscript as feature



if __name__ == '__main__':
	
	#a =  [1024, 2048, 4096, 8192, 16384]
	a=list(range(1000,20000,4))
	n_data = 2000
	i = 0

	for n in a:
				if i < n_data:
					os.system("gcc Cfiles/mvt.c -lOpenCL -lm -o Cfiles/mvt -w -fcompare-debug-second")
					os.system("./Cfiles/mvt 1 0 "+ str(n))
					i += 1

	print(i)



	#os.system("gcc Cfiles/mvt.c -lOpenCL -lm -o Cfiles/mvt -w -fcompare-debug-second")
	#os.system("./Cfiles/mvt 0 0 10 10 10 10")
	






