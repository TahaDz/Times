import os
import FeaturesExtractor.myscript as feature




if __name__ == '__main__':
	
	a = [128, 256, 512, 1024, 2048]
	i = 0
	#os.system("./Cfiles/2mm 0 0 "+ str(10)+" "+ str(4)+" "+ str(10)+" "+ str(10))
	n_data = 2000 #number of data sizes
	for ni in a:
			for nj in a:
					for nk in a:
							for nl in a:
									for nm in a :
										if i < n_data:
											os.system("gcc Cfiles/3mm.c -lOpenCL -lm -o Cfiles/3mm -w -fcompare-debug-second")
											os.system("./Cfiles/3mm 1 0 "+ str(ni)+" "+ str(nj)+" "+ str(nk)+" "+ str(nl)+" "+ str(nm))
											i += 1


	print(i)




	






