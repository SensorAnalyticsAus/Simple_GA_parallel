#define Choice 1
#define  SHORT 1
#define SunOs  0

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <mpi.h>

#define Maxpop 10000+1
#define Nvar   10+1
#define Maxchrom 16
#define Fmultiple 2
#define Tol (pow((10.),(-30.)) )

#if SunOs
#define RAND_MAX (pow((2.),(31.)) -(1.))
#endif

typedef int chromosome;

typedef struct{
              chromosome chrom[Nvar+1];
              double x[Nvar+1],objective,fitness;
              int parent1, parent2;
              int xsite;
              } ind_t;

typedef struct{
              int gen;
              chromosome chrom[Nvar+1];
              double x[Nvar+1],objective;
              } max_t;

//function signatures

void initdata(int *popsize,int *lchrom,int *maxgen,double *pcross,
      double *pmutation,unsigned long *nmutation,unsigned long *ncross,int *nvar,
      double *rvar, int rank);

double chrom_map_x(int chrom,double range1,double range2,int lchrom);

void initpop(int popsize,ind_t *oldpop,int lchrom,int nvar,double *rvar, int rank);

int flip(double probability);

double random1(void);

void bit_on(int *chrom, int i);

double objfunc(double *x,int nvar);

void statistics(int popsize,double *max,double *min,
                 double *sumfitness,ind_t *pop,max_t *max_ult,int nvar,int gen, int *jmax, int localgen);

void initreport(int popsize,int lchrom,int maxgen,double pcross,
                double pmutation,double max,double avg,double min,
                double sumfitness);

void scalepop(int popsize, double max, double avg, double min,
                              double *sumfitness,ind_t *pop);

void prescale(double umax, double uavg, double umin, double *a, double *b);

void pre_select1(int popsize, double avg, ind_t *pop, int *choices);

int select1(int *nremain,int *choices);

int  mutation(double pmutation,unsigned long *nmutation);

int  rnd(int low,int high);

int round_(int j, int lchrom);

void crossover(int *parent1,int *parent2,int *child1,int *child2,
               int lchrom,unsigned long *ncross,unsigned long *nmutation,int *jcross,
               double pcross,double pmutation, int nvar);

int ga_select(int popsize, double sumfitness, ind_t *pop);

void generation(int popsize,double *sumfitness,ind_t *oldpop,
     ind_t *newpop,int lchrom,unsigned long *ncross,unsigned long *nmutation,
     double pcross,double pmutation,int nvar, double *rvar,double avg);

int  bit_check(int chrom, int i);

int main(int argc, char *argv[])
{
  int popsize, lchrom, maxgen, gen, nvar, jmax;
  unsigned long ncross, nmutation;
  double max,min,avg,pcross,pmutation,sumfitness,rvar[2*Nvar+1];
  ind_t oldpop[Maxpop+1], newpop[Maxpop+1];
  max_t max_ult, global_max_ult;
  time_t t_start, t_finish, t_diff, r_start, r_finish, r_diff;

  //MPI variables
  int world_size, rank, rc;
  MPI_Status status;

  //Parallel grogram starts from over here
  rc = MPI_Init(&argc, &argv);

  //raise an error if exceptions occurs
  if (rc != MPI_SUCCESS)
  {
    printf("Error starting MPI program. Terminating... \n");
    MPI_Abort(MPI_COMM_WORLD, rc);
  }

  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if(rank == 0)
  {
    //start timer
    time(&t_start);
    printf("Execution has started. Please wait for the results\n");
  }

  //get the initial parameters
  initdata(&popsize,&lchrom,&maxgen,&pcross,&pmutation,&nmutation,&ncross, &nvar, rvar, rank);

  //calculate my chunk of population from the total population
  int process_pop = popsize; //   /world_size;

  //create my chunk of total population
  initpop(process_pop,oldpop,lchrom,nvar,rvar,rank);

  // since no generations have been done so far
  gen = 0;
  int localgen = 1;
  //max_ult structure contains the currenrt maximum value that has occured so far
  //it is initialised for all the processes
  //max_ult.gen = 0;
  //max_ult.objective = 0.;
  //calculate statistics for my chuck of initial population
  statistics(process_pop,&max,&min,&sumfitness,oldpop, &max_ult,nvar,gen, &jmax,localgen);

  // root process gather the statistics from all other processes
  // find the max from all the processes
  double global_max;
  MPI_Reduce(&max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  // find the min from all the processes
  double global_min;
  MPI_Reduce(&min, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

  // find the total fitness from all the processes
  double global_fitness;
  MPI_Reduce(&sumfitness, &global_fitness, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  //every process calculates its average. It is used in other generation
  avg = sumfitness/process_pop;

  //if I am rank 0 I will do the following
  if (rank == 0)
  {
    // calculate the average fitness of the group
    double global_average = global_fitness/popsize;
    //generate an initial report
    initreport(popsize,lchrom,maxgen,pcross,pmutation,global_max,global_average,global_min,global_fitness);
  }

//every process generates its part of population and send the results to root
do{

    gen+=world_size;
    localgen++;
    //scale the population
    scalepop(process_pop,max,avg,min,&sumfitness,oldpop);
    // generate new population
    generation(process_pop,&sumfitness,oldpop,newpop,lchrom,&ncross,&nmutation,pcross,pmutation,nvar,rvar,avg);
    //generate new statistics
    statistics(process_pop,&max,&min,&sumfitness,oldpop, &max_ult,nvar,gen, &jmax,localgen);
    //copy population
    memcpy(oldpop, newpop, sizeof(newpop));

  } while(gen <= maxgen);

  //once the generation process has been completed root process gather information
  //from other processes
  //root process ask for maximum fitness value from each process
  MPI_Reduce(&max_ult.objective, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  //root process broadcast value to everyone
  MPI_Bcast(&global_max, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  //processes check if their max_ult objective is equal to global max
  //if it is equal they send their data back to root
  double max_pop_data[8];
  if(max_ult.objective == global_max)
  {
    int j;
    for (j=0; j<3; j++)
      max_pop_data[j] = (double)oldpop[jmax].chrom[j];

    memcpy(&(max_pop_data[3]),&(oldpop[jmax].x[0]), sizeof(double)*(nvar));
    max_pop_data[6] = (double)max_ult.gen;
    max_pop_data[7] = (double)rank;

    MPI_Send(max_pop_data, 8, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD );
  }

  //root will start updating global_max_ult
  if(rank == 0)
  {

    //objective is the same as global max
    global_max_ult.objective = global_max;

    //i will wait for the process with maximum fitness to send me its x values and chromosomes
    //if my max is differnt than the global max
    if(max_ult.objective != global_max)
    {
      MPI_Recv(max_pop_data, 8, MPI_DOUBLE, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
      int i;
      for(i = 0; i<3; i++)
        global_max_ult.chrom[i] = (int)max_pop_data[i];
      memcpy(&(global_max_ult.x[0]),&(max_pop_data[3]), sizeof(double)*(nvar));
      global_max_ult.gen = (int)max_pop_data[6];
    }
    else //root has maximum value
    {
      memcpy(&(global_max_ult.chrom[0]),&(max_ult.chrom[0]), sizeof(chromosome)*(nvar+1));
      memcpy(&(global_max_ult.x[0]),&(max_ult.x[0]), sizeof(double)*(nvar+1));
      global_max_ult.gen = max_ult.gen;
      max_pop_data[7] = 0;
    }
      time(&t_finish);
      t_diff = t_finish-t_start;
      printf("Total Time (mins.) = %f\n", (float) t_diff/60.);

      //print the report
      printf("\nOptimal variable values for this objective function:\n");
      int i =0;
      for(i=1; i <= nvar; i++)
          printf("%5f \n",max_ult.x[i]);

      printf("\nResultant fitness:\n");
      printf("%5f \n",global_max_ult.objective);

      printf("Generated in iteration %d of process with rank %d\n",global_max_ult.gen, (int)max_pop_data[7]);

  }


  // Finalize the MPI environment.
  MPI_Finalize();
  return 0;
}

/***********************************************************************
* Function as needed by initialisation
*
************************************************************************/
/***********************************************************************
* this fuction initialise every process in the world to the initial
* parameters.
************************************************************************/
void initdata(int *popsize,int *lchrom,int *maxgen,double *pcross,
              double *pmutation,unsigned long *nmutation,unsigned long *ncross,int *nvar,
              double *rvar,int rank)
{
  register int i,j;
  FILE *fp;
  char buf[120], dummy[50];
  unsigned seed;

  if((fp = fopen("sga3.var","r")) == (FILE *)NULL)
  {printf("something wrong with sga3.var\n");exit(1);}


  fgets(buf,120,fp); sscanf(buf,"%s %d",dummy,popsize);

  fgets(buf,120,fp); sscanf(buf,"%s %d",dummy,lchrom);

  fgets(buf,120,fp); sscanf(buf,"%s %d",dummy,maxgen);

  fgets(buf,120,fp); sscanf(buf,"%s %lf",dummy,pcross);

  fgets(buf,120,fp); sscanf(buf,"%s %lf",dummy,pmutation);

  fgets(buf,120,fp); sscanf(buf,"%s %d",dummy,&seed);

  fgets(buf,120,fp); sscanf(buf,"%s %d",dummy,nvar);

  for(i = 1; i <= *nvar; i++)
  {
    j = 2*i-1;
    fgets(buf,120,fp);
    sscanf(buf,"%s %lf %s %lf",dummy,&rvar[j],dummy,&rvar[j+1]);
  }

  srand(seed + (rank | time(NULL)));

  *nmutation = 0;

  *ncross = 0;

  fclose(fp);
}

/***********************************************************************
* Every process in the world generates a random population of size
* population parameter from the file / number of processors in group.
************************************************************************/
void initpop(int popsize,ind_t *oldpop,int lchrom,int nvar,double *rvar, int rank)
{
  register int k,kk;
  int j, j1, flip_t;

  //initialise all chromosomes to 0
  for(j = 1; j <= popsize; j++)                                /* zero lchrom*/
    for(k = 1; k <= nvar; oldpop[j].chrom[k++] = 0);

  //create a random population of varibles as specified by nvar
    for(j = 1; j <= popsize; j++)
    {
        for(k = 1; k <= nvar; k++)
        {
          for(j1 = 1; j1 <= lchrom; j1++)
          {
            flip_t = flip(0.5);
            //printf("rank is %d flip is %d\n", rank, flip_t);
            if(flip_t > 0)
              bit_on(&(oldpop[j].chrom[k]), j1);
            }
        }

        //calculate the values for x depending upon the chromosomes
        for(k = 1; k <= nvar; k++)
        {
          kk = 2*k - 1;
          oldpop[j].x[k] = chrom_map_x(oldpop[j].chrom[k],
                                 rvar[kk],rvar[kk+1],lchrom);
        }

        //calculate the objective value of the initial population generated
        oldpop[j].objective = objfunc(oldpop[j].x,nvar);

        //set other varibles to zero initially
        oldpop[j].parent1 = oldpop[j].parent2= oldpop[j].xsite = 0;
    }
}

/***********************************************************************
* This function makes sure that the random value generated lies in
* the range as specified by the initial parameters.
************************************************************************/
double chrom_map_x(int chrom,double range1,double range2,int lchrom)
{
  double diff,add_percent,res;

  diff = range2-range1;
  add_percent = ((double) chrom/ (pow(2.,(double)lchrom)-1.))*diff;
  res = range1 + add_percent;
  return(res);
}

////////////

/***********************************************************************
* This function generates a random number
*
************************************************************************/
double random1(void)
{
  double result;
  result = (double) rand()/RAND_MAX;
  return(result);
}

/***********************************************************************
* Makes the bit 1 as specified by value i
*
************************************************************************/
void bit_on(int *chrom, int i)
{
  switch(i)
  {
    case 1:  *chrom |= 01; break;  /*set on the 1st bit */
    case 2:  *chrom |= 02; break;
    case 3:  *chrom |= 04; break;
    case 4:  *chrom |= 010; break;
    case 5:  *chrom |= 020; break;
    case 6:  *chrom |= 040; break;
    case 7:  *chrom |= 0100; break;
    case 8:  *chrom |= 0200; break;
    case 9:  *chrom |= 0400; break;
    case 10: *chrom |= 01000; break;
    case 11: *chrom |= 02000; break;
    case 12: *chrom |= 04000; break;
    case 13: *chrom |= 010000; break;
    case 14: *chrom |= 020000; break;
    case 15: *chrom |= 040000; break;
    case 16: *chrom |= 0100000; break;
    case 17: *chrom |= 0200000; break;
    case 18: *chrom |= 0400000; break;
    case 19: *chrom |= 01000000; break;
    case 20: *chrom |= 02000000; break;
    case 21: *chrom |= 04000000; break;
    case 22: *chrom |= 010000000; break;
    case 23: *chrom |= 020000000; break;
    case 24: *chrom |= 040000000; break;
    case 25: *chrom |= 0100000000; break;
    case 26: *chrom |= 0200000000; break;
    case 27: *chrom |= 0400000000; break;
    case 28: *chrom |= 01000000000; break;
    case 29: *chrom |= 02000000000; break;
    case 30: *chrom |= 04000000000; break;
    case 31: *chrom |= 010000000000; break;
  }
}

/***********************************************************************
* Statistics function for processes other than the root process
*
************************************************************************/

void statistics(int popsize,double *max, double *min,
                 double *sumfitness,ind_t *pop,max_t *max_ult,int nvar,int gen, int *jmax, int localgen)
{
  int j;

  *jmax = 1;

  *sumfitness = pop[1].objective;
  *min        = pop[1].objective;
  *max        = pop[1].objective;

  for(j = 2; j <= popsize; j++)
  {
    *sumfitness += pop[j].objective;
    if(pop[j].objective > *max)
    {
      *max = pop[j].objective;
      *jmax = j;
    }

    if(pop[j].objective < *min)
      *min = pop[j].objective;
  }

  //i need this over here because I want to save the chromosomes and x values
  //for the max
  if(*max > max_ult->objective)
  {
     max_ult->objective = *max;
     max_ult->gen = localgen;
     memcpy(&(max_ult->chrom[0]),&(pop[*jmax].chrom[0]), sizeof(chromosome)*(nvar+1));
     memcpy(&(max_ult->x[0]),&(pop[*jmax].x[0]), sizeof(double)*(nvar+1));
  }
}

/***********************************************************************
* Used by root to generate an initial report and save it in a file.
*
************************************************************************/

void initreport(int popsize,int lchrom,int maxgen,double pcross,
                double pmutation,double max,double avg,double min,
                double sumfitness)
{
  FILE *fpout;

  if((fpout=fopen("genout.dat","w")) == (FILE *) NULL)
  {printf("cannot open genout.dat\n");exit(1);}

  fprintf(fpout,"Population size (popsize) = %5d\n",popsize);
  fprintf(fpout,"Chromosome length (lchrom) = %5d\n",lchrom);
  fprintf(fpout,"Maximum # of generations (maxgen) %5d\n",maxgen);
  fprintf(fpout,"Crossover probability (pcross) = %10.5e\n",pcross);
  fprintf(fpout,"Mutation probability (pmutation) = %10.5e\n",pmutation);
  fprintf(fpout,"\n Initial Generation Statistics\n");
  fprintf(fpout,"---------------------------------\n");
  fprintf(fpout,"\n");
  fprintf(fpout,"Initial population maximum fitness = %10.5e\n", max);
  fprintf(fpout,"Initial population average fitness = %10.5e\n",avg);
  fprintf(fpout,"Initial population minimum fitness = %10.5e\n",min);
  fprintf(fpout,"Initial population sum of  fitness = %10.5e\n",sumfitness);
  fprintf(fpout,"\n\n\n");
  fclose(fpout);
}
/***********************************************************************
* Functions as needed by scaling
*
************************************************************************/

/***********************************************************************
* Scale the population to make sure there are a variety of fitnesses in
* the population
************************************************************************/
void scalepop(int popsize, double max, double avg, double min,
              double *sumfitness,ind_t *pop)
{
  register int j;
  double a, b;

  prescale(max,avg,min,&a,&b);
  *sumfitness = 0.;

  for(j = 1; j <= popsize; j++)
  {
    pop[j].fitness = a*pop[j].objective+b;
    *sumfitness += pop[j].fitness;
  }
}

/***********************************************************************
* Used by scale population to prescale fitnesses
*
************************************************************************/
void prescale(double umax, double uavg, double umin, double *a, double *b)
{
  double delta;

  if(umin > (Fmultiple*uavg-umax)/(Fmultiple-1.)) /*Non-neg test*/
  {
  delta = umax-uavg;
  if(delta == 0.) delta = .00000001;
  *a = (Fmultiple-1.)*uavg/delta;
  *b = uavg*(umax-Fmultiple*uavg)/delta;
  }
  else
  {
  delta = uavg - umin;
  if(delta == 0.) delta = .00000001;
  *a = uavg/delta;
  *b = -umin*uavg/delta;
  }
}

/***********************************************************************
* Functions as needed by the generation
*
************************************************************************/
/***********************************************************************
* on or off the bit as specified by i
*
************************************************************************/

void bit_on_off(int *chrom, int i)
{
  switch(i)
  {
    case 1:  *chrom ^= 01; break;  /*set on the 1st bit */
    case 2:  *chrom ^= 02; break;
    case 3:  *chrom ^= 04; break;
    case 4:  *chrom ^= 010; break;
    case 5:  *chrom ^= 020; break;
    case 6:  *chrom ^= 040; break;
    case 7:  *chrom ^= 0100; break;
    case 8:  *chrom ^= 0200; break;
    case 9:  *chrom ^= 0400; break;
    case 10: *chrom ^= 01000; break;
    case 11: *chrom ^= 02000; break;
    case 12: *chrom ^= 04000; break;
    case 13: *chrom ^= 010000; break;
    case 14: *chrom ^= 020000; break;
    case 15: *chrom ^= 040000; break;
    case 16: *chrom ^= 0100000; break;
    case 17: *chrom ^= 0200000; break;
    case 18: *chrom ^= 0400000; break;
    case 19: *chrom ^= 01000000; break;
    case 20: *chrom ^= 02000000; break;
    case 21: *chrom ^= 04000000; break;
    case 22: *chrom ^= 010000000; break;
    case 23: *chrom ^= 020000000; break;
    case 24: *chrom ^= 040000000; break;
    case 25: *chrom ^= 0100000000; break;
    case 26: *chrom ^= 0200000000; break;
    case 27: *chrom ^= 0400000000; break;
    case 28: *chrom ^= 01000000000; break;
    case 29: *chrom ^= 02000000000; break;
    case 30: *chrom ^= 04000000000; break;
    case 31: *chrom ^= 010000000000; break;
  }
}
/***********************************************************************
* If the bit specified by i is 1 return 1 else return 0
*
************************************************************************/

int  bit_check(int chrom, int i)
{
  switch(i)
  {
    case 1:  if(chrom & 01)return(1); break;
    case 2:  if(chrom & 02)return(1); break;
    case 3:  if(chrom & 04)return(1); break;
    case 4:  if(chrom & 010)return(1); break;
    case 5:  if(chrom & 020)return(1); break;
    case 6:  if(chrom & 040)return(1); break;
    case 7:  if(chrom & 0100)return(1); break;
    case 8:  if(chrom & 0200)return(1); break;
    case 9:  if(chrom & 0400)return(1); break;
    case 10: if(chrom & 01000)return(1); break;
    case 11: if(chrom & 02000)return(1); break;
    case 12: if(chrom & 04000)return(1); break;
    case 13: if(chrom & 010000)return(1); break;
    case 14: if(chrom & 020000)return(1); break;
    case 15: if(chrom & 040000)return(1); break;
    case 16: if(chrom & 0100000)return(1); break;
    case 17: if(chrom & 0200000)return(1); break;
    case 18: if(chrom & 0400000)return(1); break;
    case 19: if(chrom & 01000000)return(1); break;
    case 20: if(chrom & 02000000)return(1); break;
    case 21: if(chrom & 04000000)return(1); break;
    case 22: if(chrom & 010000000)return(1); break;
    case 23: if(chrom & 020000000)return(1); break;
    case 24: if(chrom & 040000000)return(1); break;
    case 25: if(chrom & 0100000000)return(1); break;
    case 26: if(chrom & 0200000000)return(1); break;
    case 27: if(chrom & 0400000000)return(1); break;
    case 28: if(chrom & 01000000000)return(1); break;
    case 29: if(chrom & 02000000000)return(1); break;
    case 30: if(chrom & 04000000000)return(1); break;
    case 31: if(chrom & 010000000000)return(1); break;
  }
  return(0);
}

/***********************************************************************
* Generate new population by randomly selecting chromosomes from
* the population and doing their crossover
************************************************************************/

void generation(int popsize,double *sumfitness,ind_t *oldpop,
     ind_t *newpop,int lchrom,unsigned long *ncross,unsigned long *nmutation,
     double pcross,double pmutation,int nvar, double *rvar,double avg)
{
  register int k, kk;
  int j, mate1, mate2, jcross,nremain, choices[Maxpop+1];
  j = 1;
  //printf("%f\n", avg);
  #if Choice
    nremain = popsize;
    pre_select1(popsize, avg, oldpop, choices);
  #endif
  //select mates
  do{
    #if Choice
      mate1 = select1(&nremain, choices);
      mate2 = select1(&nremain, choices);
    #else
      mate1 = ga_select(popsize, *sumfitness, oldpop);
      mate2 = ga_select(popsize, *sumfitness, oldpop);
    #endif
    //printf("i am here 8\n");
    //do their crossover
    crossover(oldpop[mate1].chrom, oldpop[mate2].chrom,
          &(newpop[j].chrom[0]), &(newpop[j+1].chrom[0]),
          lchrom,ncross,nmutation,&jcross,pcross,pmutation,nvar);


    //calculate the fintess and x of new population
    for(k = 1; k <= nvar; k++)
    {
      kk = 2*k - 1;
      newpop[j].x[k] = chrom_map_x(newpop[j].chrom[k],rvar[kk],rvar[kk+1],lchrom);
    }

    newpop[j].objective = objfunc(newpop[j].x, nvar);
    newpop[j].parent1 = mate1;
    newpop[j].parent2 = mate2;
    newpop[j].xsite = jcross;

    for(k = 1; k <= nvar; k++)
    {
      kk = 2*k - 1;
      newpop[j+1].x[k] = chrom_map_x(newpop[j+1].chrom[k],rvar[kk],rvar[kk+1],lchrom);
    }
    newpop[j+1].objective = objfunc(newpop[j+1].x, nvar);
    newpop[j+1].parent1 = mate1;
    newpop[j+1].parent2 = mate2;
    newpop[j+1].xsite = jcross;

    j += 2;

  }while(j < popsize);
}

/***********************************************************************
* Select a chromosomes based upon its fitness
*
************************************************************************/
int ga_select(int popsize, double sumfitness, ind_t *pop)
{
  int j;
  double partsum, randx;

  partsum = 0.; j = 0;
  randx = random1()*sumfitness;
  do{
      j += 1;
      partsum += pop[j].fitness;
    }while(partsum <= randx && j != popsize);
  return(j);
}

/***********************************************************************
* Crossover the two chromosomes
*
************************************************************************/
void crossover(int *parent1,int *parent2,int *child1,int *child2,
               int lchrom,unsigned long *ncross,unsigned long *nmutation,int *jcross,
               double pcross,double pmutation, int nvar)
{
  register int k,kk;
  int j, lighted, test, rn;

  memcpy(child1, parent1, sizeof(int)*(nvar+1));
  memcpy(child2, parent2, sizeof(int)*(nvar+1));

  if(flip(pcross)  == 1)
  {
    *jcross = rnd(1, nvar*lchrom-1);
    *ncross += 1;
  }
  else
    *jcross = nvar*lchrom;

  rn = 0;                                                   /*chrom counter*/
  kk = 1;
  for(j = 1; j <= *jcross; j++)
  {
    if(rn == 1)
      kk++;
    rn = round_(j, lchrom);
    k = j - (kk-1)*lchrom;
    test = mutation(pmutation, nmutation);

    /*test = [0] no change , test = 1 bit changed kth bit is altered*/
    if(test == 1)bit_on_off(&child1[kk],k);   /* mutation */
    test = mutation(pmutation, nmutation);
     /*test = [0] no change , test = 1 bit changed kth bit is altered*/
    if(test == 1)bit_on_off(&child2[kk],k); /* mutation */
    k++;
  }

  if(*jcross != nvar*lchrom)
  {
   for(j = *jcross+1; j <= nvar*lchrom; j++)
   {
     if(rn == 1) kk++;
     rn = round_(j, lchrom);
     k = j - (kk-1)*lchrom;
     lighted = bit_check(parent2[kk],k);        /*lighted = [1] if bit is on */
     test = mutation(pmutation, nmutation);
     /*test = [0] no change , test = 1 bit changed jth bit is altered*/
     bit_on(&child1[kk],k);
     if(lighted == 0) bit_on_off(&child1[kk],k);
     if(test == 1)bit_on_off(&child1[kk],k);    /* mutate */
     lighted = bit_check(parent1[kk],k);        /*lighted = [1] if bit is on */
     test = mutation(pmutation, nmutation);
     /*test = [0] no change , test = 1 bit changed jth bit is altered*/
     bit_on(&child2[kk],k);
     if(lighted == 0) bit_on_off(&child2[kk],k);
     if(test == 1)bit_on_off(&child2[kk],k);    /* mutate */
   }
  }
}

/***********************************************************************
* round function
*
************************************************************************/
int round_(int j, int lchrom)
{
  if(fmod(j,lchrom) == 0)
    return(1);
  return(0);
}

/***********************************************************************
* random function
*
************************************************************************/
int  rnd(int low,int high)
{
  int i;

  if(low >= high) i = low;
  else
  {
    i = (int) (random1()*(high-low+1) + low);
    if(i > high) i = high;
  }
  return(i);
}

/***********************************************************************
* mutate the chrome
*
************************************************************************/
int  mutation(double pmutation,unsigned long *nmutation)
{
  int mutate;
  mutate = flip(pmutation);
  if(mutate == 1)
  {
    *nmutation += 1;
    return(1);
  }
  else
    return(0);
}

/***********************************************************************
* selecting chromes for generation
*
************************************************************************/
void pre_select1(int popsize, double avg, ind_t *pop, int *choices)
{
  register int j, k;
  int jassign, winner;
  double expected, fraction[Maxpop+1], whole;
  //printf("%f\n", avg);
  j = 0; k = 0;

  do{
    j++;
    expected = pop[j].fitness/(avg+1.e-30);
    fraction[j] = modf(expected, &whole);
    jassign = (int) whole;
    while(jassign > 0)
    {
      k++;
      jassign--;
      choices[k] = j;
    }
  }while(j != popsize);

  j = 0;
  while(k < popsize)
  {
    j++;
    if(j > popsize)
      j = 1;
    if(fraction[j] > 0.0)
    {
      winner = flip(fraction[j]);
      if(winner == 1)
      {
        k++;
        choices[k] = j;
        (fraction[j])--;
      }
    }
  }
  //printf("i am going out\n");
}

/***********************************************************************
* selecting chrome for generation
*
************************************************************************/
int select1(int *nremain,int *choices)
{
  int jpick, index;
  jpick = rnd(1, *nremain);

  index = choices[jpick];

  choices[jpick] = choices[*nremain];

  *nremain = *nremain - 1;

  return(index);
}

/***********************************************************************
* This function generates 0 or 1 with 50 % probability
*
************************************************************************/

int flip(double probability)
{
  double random1_t;
  if(probability == 1.0)
    return(1);

  random1_t = random1();

  if(random1_t <= probability)
    return(1);
  else
    return(0);
}
