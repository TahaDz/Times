import os
import FeaturesExtractor.myscript as feature



if __name__ == '__main__':
	
	#a = [1024, 2048, 4096 , 8192, 16384]
	a = list(range(1000,17000,258))
	l = 0
	n_data = 2000
	for i in range(0,len(a)-1):
		ni = a[i]

		for j in range(i,len(a)):
			nj = a[j]

			if l < n_data:
				os.system("gcc Cfiles/2DConvolution.c -lOpenCL -lm -o Cfiles/2DConvolution -w -fcompare-debug-second")
				os.system("./Cfiles/2DConvolution 1 0 "+ str(ni)+" "+ str(nj))
				l += 1

	print(l)



	#os.system("gcc Cfiles/2mm.c -lOpenCL -lm -o Cfiles/2mm -w -fcompare-debug-second")
	#os.system("./Cfiles/2mm 0 0 10 10 10 10")
	






