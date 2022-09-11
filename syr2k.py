import os





if __name__ == '__main__':
	
	#a = [256, 512, 1024, 2048, 4096]
	i = 0
	a = list(range(100,2100,1))
	n_data = 2000
	for ni in a:
		if i < n_data:
					os.system("gcc Cfiles/syr2k.c -lOpenCL -lm -o Cfiles/syr2k -w -fcompare-debug-second")
					os.system("./Cfiles/syr2k 0 0 "+ str(ni)+" "+ str(ni))

					i += 1

	print(i)



	#os.system("gcc Cfiles/syr2k.c -lOpenCL -lm -o Cfiles/syr2k -w -fcompare-debug-second")
	#os.system("./Cfiles/syr2k 0 0 10 10 10 10")
	






