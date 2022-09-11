import os






if __name__ == '__main__':
	
	a = list(range(512,8192,123))
	l = 0
	n_data = 2000
	for i in range(0,len(a)-1):
		ni = a[i]

		for j in range(i,len(a)):
			nj = a[j]

			if l < n_data:
				os.system("gcc Cfiles/covariance.c -lOpenCL -lm -o Cfiles/covariance -w -fcompare-debug-second")
				os.system("./Cfiles/covariance 0 0 "+ str(ni)+" "+ str(nj))
				l += 1

	print(l)



	#os.system("gcc Cfiles/2mm.c -lOpenCL -lm -o Cfiles/2mm -w -fcompare-debug-second")
	#os.system("./Cfiles/2mm 0 0 10 10 10 10")
	






