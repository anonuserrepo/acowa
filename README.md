This repository contains implementations of naive averaging, OWA, ACOWA, debiased averaging, CSL, and DANE.
The implementations are written in C++ and use the Armadillo, ensmallen, and liblbfgs libraries.
Armadillo and ensmallen are packaged with this repository, but you will need to install some dependencies.

If using Ubuntu or Debian, install dependencies with:

```
sudo apt-get install liblbfgs-dev libopenblas-dev g++ make
```

or, if on OS X, make sure that a C++ compiler is installed and run

```
brew install liblbfgs openblas
```

Once dependencies are installed, the following two targets are available:

 * `make single`: build programs to test ACOWA and competitors in the multicore single-node setting
 * `make mpi`: build MPI-enabled programs for distributed usage

Each program will print the required options.
Instead of running once on a given dataset, each program (e.g. `sweep_acowa`) will sweep a variety of lambda values on the same dataset.
In this way, performance for different numbers of nonzeros can be given.

For example, try running naive averaging, OWA, and ACOWA on the included newsgroups dataset:

```
./sweep_naive_avg data/newsgroups.svm naive_avg.csv 1 -6 1 21 256
./sweep_owa data/newsgroups.svm owa.csv 1 -6 1 21 256
./sweep_acowa data/newsgroups.svm acowa.csv 1 -6 1 21 256
```

Other datasets should be in libsvm format.
