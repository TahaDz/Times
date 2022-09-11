import os



if __name__ == '__main__':

    i = 0
    a = list(range(500,4500,2))
    n_data = 2000
    for ni in a:
        if i < n_data:
            os.system("gcc Cfiles/jacobi2D.c -lOpenCL -lm -o Cfiles/jacobi2D -w -fcompare-debug-second")
            os.system("./Cfiles/jacobi2D 1 0 "+ str(ni))
            i += 1

    print(i)


