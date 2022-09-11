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
	
	#a = [256, 512, 1024, 2048, 4096]
	i = 0
	a = list(range(100,2100,1))
	n_data = 2000
	for ni in a:
		if i < n_data:
					os.system("gcc Cfiles/syr2k.c -lOpenCL -lm -o Cfiles/syr2k -w -fcompare-debug-second")
					os.system("./Cfiles/syr2k 1 0 "+ str(ni)+" "+ str(ni))

					i += 1

	print(i)



	#os.system("gcc Cfiles/syr2k.c -lOpenCL -lm -o Cfiles/syr2k -w -fcompare-debug-second")
	#os.system("./Cfiles/syr2k 0 0 10 10 10 10")
	






