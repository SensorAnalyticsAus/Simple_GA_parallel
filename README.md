## Prerequisite
##### Install Message Passing Interface (MPI) 
````
sudo apt-get install openmpi-bin openmpi-common libopenmpi-dev libgtk2.0-dev openmpi-doc
````
##### Or build from either of these sources: <br> <p><p> https://www.mpich.org/downloads/ <p> https://www.open-mpi.org/software/ompi/v4.1/
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
## Files
##### SGA parameters are set in *sga3.var*. Explanation of these parameters can be found in Simple Genetic Algorithm Explainer.pdf [Section 4]. The objective function is written in *obj3.c*.
## Acknowledgements
##### This C code is based upon D.E. Goldberg's Genetic Algorithms in Search, Optimisation and Machine Learning 1989. MPI parallelisation by N. Zaidi 2016.

##### ©SAA 2022
