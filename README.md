## Prequisite
##### Install Message Passing Interface (MPI) from: <br> <p><p> https://www.mpich.org/downloads/ <p> or <p> https://www.open-mpi.org/software/ompi/v4.1/
## Compile
Run the following command in the downloaded folder:
<pre><code>
make
</code></pre>
## Run
##### For quad core processor.
<pre><code>
mpirun -np 4 ./sga_parr
</code></pre>
##### The number of processors, *-np*, should be set equal to the number of cores of your CPU.
## Acknowledgements
##### This C code is based upon D.E. Goldberg's Genetic Algorithms in Search, Optimisation and Machine Learning. MPI parallelisation by N. Zaidi.


