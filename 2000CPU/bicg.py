import os
import FeaturesExtractor.myscript as feature




if __name__ == '__main__':
	
	#a = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
	a = list(range(100,100100,50))
	n_data = 2000

	i = 0
	#os.system("./Cfiles/2mm 0 0 "+ str(10)+" "+ str(4)+" "+ str(10)+" "+ str(10))
	for nx in a:
		for ny in a:
			if i < n_data:
				os.system("gcc Cfiles/bicg.c -lOpenCL -lm -o Cfiles/bicg -w -fcompare-debug-second")
				os.system("./Cfiles/bicg 1 0 "+ str(nx)+" "+ str(ny))
				i += 1

	print(i)




	






