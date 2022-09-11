import os




if __name__ == '__main__':
	
	#a = [4,16,32,64,128, 256, 512, 1024, 2048, 4096,8192]
	a = list(range(1000,17000,255))
	i = 0
	n_data = 2000
	for index,wa in enumerate(a):
		for ha in a[index:]:
					if i < n_data:
						os.system("gcc Cfiles/transpose.c -lOpenCL -lm -o Cfiles/transpose -w -fcompare-debug-second")
						os.system("./Cfiles/transpose 0 0 "+ str(wa)+" "+ str(ha))
						i += 1

	print(i)




	






