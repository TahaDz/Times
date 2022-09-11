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
				os.system("gcc Cfiles/correlation.c -lOpenCL -lm -o Cfiles/correlation -w -fcompare-debug-second")
				os.system("./Cfiles/correlation 0 0 "+ str(ni)+" "+ str(nj))
				l += 1

	print(l)









