import os
import FeaturesExtractor.myscript as feature






## extract program 'pg' features
def feature_extraction (pg):
	print(pg)
	feature.extract_features(pg)
	
	#os.system("python3 FeaturesExtractor/myscript.py Cfiles/"+pg) # .cl files (kernels)
	#os.system("cp Features/b1.txt Features/ProgramFeatures/"+pg[:-2]+".txt")

	
	
	

def import_pgs():
	
	### List all files in the directory

	file_list=sorted(os.listdir("Cfiles"))
	#print (file_list)	

    
	### List only C program
	c_files = [] ## List of [program, suitable device (platform, device), order in queue]
	for f in file_list :

		if f.endswith(".c" ) :
			######## name of the pg
			c_files.append(f[:-2])
		
		
	
	#print(c_files)

	#print('================')
	
	return c_files
	
if __name__ == '__main__':
	
	c_files = import_pgs()
	#print(c_files)

	for pg in c_files :
		feature_extraction(pg)
		#time_measurment(pg)
	






