default: example_knl

CC = mpiicpc
CFLAGS = -O3 -qopenmp -mkl -march=native

example_knl: main.cpp func.cpp check_func.cpp func.h
	$(CC) $(CFLAGS) $(CLIBS) main.cpp func.cpp check_func.cpp -o example_knl

run: example_knl
	#mpirun -np 1 ./example /apps/hw3/matrix/torso3.csr
	#srun --exclusive  -p cpu ./example /apps/hw3/matrix/torso3.csr
	./example_knl /apps/hw3/matrix/torso3.csr
	#srun -n 1 ./example /apps/hw3/matrix/torso3.csr

mrun: example_knl
	numactl -m 1 ./example_knl /apps/hw3/matrix/torso3.csr
clean:
	rm -rf ./example
