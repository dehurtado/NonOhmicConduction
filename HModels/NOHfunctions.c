#include <math.h>
#include <stdlib.h>
#include <stdio.h>

void computeRates(double VOI, double* CONSTANTS, double* RATES, double* STATES,
     double* ALGEBRAIC);
     

void update(double* ua_old, double* w, double* CONSTANTS, double VOI, 
            double dt, int ndofs,  double* fa){

    double* RATES;  RATES = (double*)calloc(8, sizeof(double)) ;
    double* STATES;  STATES = (double*)calloc(8, sizeof(double)) ;
    double* ALGEBRAIC;  ALGEBRAIC = (double*)calloc(25, sizeof(double)) ; 

    for (int i = 0; i<ndofs; i++){ // Loop over nodes
        STATES[0] = ua_old[i];

        for (int j = 0; j<7; j++){ // Loop over state variables
            STATES[j+1] = w[7*i+j];
        }

        computeRates(VOI, CONSTANTS, RATES, STATES, ALGEBRAIC);

//  Update gating variables
        for (int j = 0; j<7; j++){ // Loop over state variables
            w[7*i+j] = w[7*i+j] + dt*RATES[j+1];
        }

//  Save force value
        fa[i] = RATES[0];

    }
    free(RATES);    free(STATES);    free(ALGEBRAIC);


}

double linear_interpolation(double y, double* yarr, double* sarr, 
                            double* yrange, double* srange){
    
    int i;
    double sig;

    sig = 0.0;
    
    double ymin = yrange[0]; 
    double ymax = yrange[1];
    double smin = srange[0];
    double smax = srange[1];

//    printf("%f ", smin);
//    printf("%f ", smax);
    if (y < ymin){
        sig = smin;    
    } else if (y > ymax){
        sig = smax;
    }else{
        i = 0;
        while(yarr[i] < ymax){
            if (yarr[i] < y && y <= yarr[i+1]){
                sig = sarr[i] + (y - yarr[i])*(sarr[i+1] - sarr[i])/(yarr[i+1] - yarr[i]);
            }
            i++;
        }
    }

    return sig;

}


void effcond(double* gradu, double* y_arr, double* s_arr, double* y_range,
             double* s_range, int npoints, double* sig){

    for (int i = 0; i<npoints; i++){
//        printf("%f ", gradu[i]);
        sig[i] = linear_interpolation(gradu[i], y_arr, s_arr, y_range,
                                      s_range);
//        printf(" ");
//        printf("%f ", sig[i]);
//        printf("\n");
    }
//        printf("\n");

}

















