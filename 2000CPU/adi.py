import os






if __name__ == '__main__':
	

	i = 0
	a = list(range(256,4300,2))
	n_data = 2000
	for ni in a:
		if i < n_data:
					os.system("gcc Cfiles/adi.c -lOpenCL -lm -o Cfiles/adi -w -fcompare-debug-second")
					os.system("./Cfiles/adi 1 0 "+ str(ni))

					i += 1

	print(i)



	#os.system("gcc Cfiles/2mm.c -lOpenCL -lm -o Cfiles/2mm -w -fcompare-debug-second")
	#os.system("./Cfiles/2mm 0 0 10 10 10 10")
	






