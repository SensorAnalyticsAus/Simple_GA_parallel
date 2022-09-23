/* Copyright A.I. Khan 2015 */
/*                         obj3.c */
#define Tol 1.e-30;

#include <math.h>

double    objfunc(double *x, int nvar)
{
register int i,j;
double res, result,penalty;

/* We are searching for  values x1 to x3 which produce
   a maximum value of close to 700 and add up to zero for a 
   specified function (below).
 
So we will  maximize:
f(x1,x2,x3) = 700 - (x1+x2^2+x2*x3)
subject to: x1 + x2 + x3 = 0
*/


result = 700 - (x[1]+pow(x[2],2)+x[2]*x[3]);

penalty =1000*pow(x[1]+x[2]+x[3],2);

result -= penalty;

if(fabs(result) <= 0. ) result = Tol;

return(result);
}

