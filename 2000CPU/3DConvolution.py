import os
import FeaturesExtractor.myscript as feature


if __name__ == '__main__':
	
	a = list(range(100,600,23))
	i = 0
	n_data = 2000




	l = 0
	for i in range(0,len(a)):
		ni = a[i]
		for j in range(i,len(a)):
			nj = a[j]

			for k in range(j, len(a)):
				nk = a[k]
				if l < n_data:
					os.system("gcc Cfiles/3DConvolution.c -lOpenCL -lm -o Cfiles/3DConvolution -w -fcompare-debug-second")
					os.system("./Cfiles/3DConvolution 1 0 "+ str(ni)+" "+ str(nj)+" "+ str(nk))
					l += 1


	print(l)




	






