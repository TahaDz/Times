import os






if __name__ == '__main__':
	
	#a = [128, 256, 512, 1024, 2048, 4096]
	a = list(range(100,2100,1))
	i = 0
	n_data = 2000
	for ni in a:
		if i < n_data:
			os.system("gcc Cfiles/syrk.c -lOpenCL -lm -o Cfiles/syrk -w -fcompare-debug-second")
			os.system("./Cfiles/syrk 0 0 "+ str(ni)+" "+ str(ni))
			i += 1

	print(i)



	#os.system("gcc Cfiles/syrk.c -lOpenCL -lm -o Cfiles/syrk -w -fcompare-debug-second")
	#os.system("./Cfiles/syrk 0 0 10 10 10 10")
	






