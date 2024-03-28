# You may need to modify things below to compile, if your system does not have liblbfgs on the compiler search path,
# or if the compiler itself is not on the $PATH.

#CXX = /opt/homebrew/bin/g++-12
CXXFLAGS += -Ideps/include/ -O3 -march=native -DNDEBUG -DARMA_NO_DEBUG -fopenmp

LDFLAGS += -lopenblas

# These may be necessary.
LDFLAGS += -lquadmath
LDFLAGS += -llbfgs

all: single mpi
	
single: sweep_lr sweep_naive_avg sweep_owa sweep_debias_avg sweep_dane sweep_csl sweep_acowa
mpi: sweep_mpi_naive_avg sweep_mpi_owa sweep_mpi_acowa sweep_mpi_csl sweep_mpi_dane sweep_mpi_debias_avg

# Single-node targets
sweep_naive_avg: src/sweep_naive_avg.o
	$(CXX) $(CXXFLAGS) -o $@ src/sweep_naive_avg.o $(LDFLAGS)

sweep_owa: src/sweep_owa.o
	$(CXX) $(CXXFLAGS) -o $@ src/sweep_owa.o $(LDFLAGS)

sweep_debias_avg: src/sweep_debias_avg.o
	$(CXX) $(CXXFLAGS) -o $@ src/sweep_debias_avg.o $(LDFLAGS)

sweep_lr: src/sweep_lr.o
	$(CXX) $(CXXFLAGS) -o $@ src/sweep_lr.o $(LDFLAGS)

sweep_acowa: src/sweep_acowa.o
	$(CXX) $(CXXFLAGS) -o $@ src/sweep_acowa.o $(LDFLAGS)

sweep_dane: src/sweep_dane.o
	$(CXX) $(CXXFLAGS) -o $@ src/sweep_dane.o $(LDFLAGS)

sweep_csl: src/sweep_csl.o
	$(CXX) $(CXXFLAGS) -o $@ src/sweep_csl.o $(LDFLAGS)

# MPI targets
sweep_mpi_naive_avg:
	mpicxx $(CXXFLAGS) -o $@ src/mpi/sweep_mpi_naive_avg.cpp $(LDFLAGS)

sweep_mpi_owa:
	mpicxx $(CXXFLAGS) -o $@ src/mpi/sweep_mpi_owa.cpp $(LDFLAGS)

sweep_mpi_acowa:
	mpicxx $(CXXFLAGS) -o $@ src/mpi/sweep_mpi_acowa.cpp $(LDFLAGS)

sweep_mpi_csl:
	mpicxx $(CXXFLAGS) -o $@ src/mpi/sweep_mpi_csl.cpp $(LDFLAGS)

sweep_mpi_dane:
	mpicxx $(CXXFLAGS) -o $@ src/mpi/sweep_mpi_dane.cpp $(LDFLAGS)

sweep_mpi_debias_avg:
	mpicxx $(CXXFLAGS) -o $@ src/mpi/sweep_mpi_debias_avg.cpp $(LDFLAGS)

clean:
	rm -f src/*.o sweep_*
