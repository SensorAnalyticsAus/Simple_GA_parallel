## Prerequisite
##### Install Message Passing Interface (MPI) 
````
sudo apt-get install openmpi-bin openmpi-common libopenmpi-dev openmpi-doc
````
##### Or build from either of these sources: <br> <p><p> https://www.mpich.org/downloads/ <p> https://www.open-mpi.org/software/ompi/v4.1/
## Compile
````
git clone https://github.com/SensorAnalyticsAus/Simple_GA_parallel.git
cd Simple_GA_parallel/
make
make clean
````
## Run
##### For quad core processor.
<pre><code>
mpirun -np 4 ./sga_parr
</code></pre>
##### The number of processors, *-np*, should be set equal to the number of cores of your CPU.
## Files
##### SGA parameters are set in *sga3.var*. Explanation of these parameters can be found in [Simple Genetic Algorithm Explainer - Section 4](/assets/pdf/gene.pdf). The objective function is written in *obj3.c*.
## Acknowledgements
##### This C code is based upon D.E. Goldberg's Genetic Algorithms in Search, Optimisation and Machine Learning 1989. MPI parallelisation by N. Zaidi 2016.