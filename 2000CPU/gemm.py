import os
import FeaturesExtractor.myscript as feature

def time_measurment(pg):
	output = pg
	pg = pg+'.c'
	#print("gcc Cfiles/"+pg+" -l OpenCL -o Cfiles/"+output+" -w -fcompare-debug-second")
	os.system("gcc Cfiles/"+pg+" -l OpenCL -o Cfiles/"+output+" -w -fcompare-debug-second")#"-w", "-fcompare-debug-second" to disable warnings and notes
			
	# run OpenCL files
	#print("--------- This is "+name+" ------------")
	os.system("./Cfiles/"+output+" 0 0")




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
					os.system("./Cfiles/gemm 1 0 "+ str(ni)+" "+ str(nj)+" "+ str(nk))
					l += 1


	print(l)






	






