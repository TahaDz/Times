import os
import FeaturesExtractor.myscript as feature




if __name__ == '__main__':
	
	a = [128, 256, 512, 1024, 2048, 4096]
	i = 0
	n_data = 2000
	for ni in a:
		for nj in a:
			for nk in a:
				for nl in a:
					if i < n_data:
						os.system("gcc Cfiles/2mm.c -lOpenCL -lm -o Cfiles/2mm -w -fcompare-debug-second")
						os.system("./Cfiles/2mm 1 0 "+ str(ni)+" "+ str(nj)+" "+ str(nk)+" "+ str(nl))
						i += 1

	print(i)










