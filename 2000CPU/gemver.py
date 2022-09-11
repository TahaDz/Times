import os
import FeaturesExtractor.myscript as feature



if __name__ == '__main__':
	
	#a =  [1024, 2048, 4096, 8192, 16384]
	#a = list(range(1000,11000,100))
	a = list(range(1000,11000,5))
	n_data = 2000
	i = 0

	for n in a:
			if i < n_data:
					os.system("gcc Cfiles/gemver.c -lOpenCL -lm -o Cfiles/gemver -w -fcompare-debug-second")
					os.system("./Cfiles/gemver 1 0 "+ str(n))

					i += 1

	print(i)








