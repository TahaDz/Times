import os




if __name__ == '__main__':
	
	#a = [128, 256, 512, 1024, 2048]
	a = list(range(100,2050,90))
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
					os.system("gcc Cfiles/gemm.c -lOpenCL -lm -o Cfiles/gemm -w -fcompare-debug-second")
					os.system("./Cfiles/gemm 0 0 "+ str(ni)+" "+ str(nj)+" "+ str(nk))
					l += 1


	print(l)






	






