#include <math.h>
#include <stdlib.h>
#include <stdio.h>


void computeRates(double VOI, double* CONSTANTS, double* RATES, double* STATES,
     double* ALGEBRAIC);
     

double finiteDifference(int i, double* V,  double* gamma){
    double fdif = 0;
    
    fdif =  gamma[i-1]*(V[i-1]-V[i]) - gamma[i]*(V[i]-V[i+1]) ;
   
    return fdif;

}

     
void FDupdate(double* v_old, double* w_old, double* CONSTANTS, double VOI, 
     double dt, int n, int nstates, int nalg, double* gamma, double source, 
     int rev, double* v_new, double* w_new){
    
    double* RATES;  RATES = (double*)calloc(nstates, sizeof(double)) ;
    double* STATES;  STATES = (double*)calloc(nstates, sizeof(double)) ;
    double* ALGEBRAIC;  ALGEBRAIC = (double*)calloc(nalg, sizeof(double)) ; 
    double* dv; dv = (double*)calloc(n, sizeof(double)) ;

    double fdif = 0;
    int ngates = nstates - 1;

//  Assign states in each node and compute rates
    for (int i = 1; i<n-1; i++){ // Loop over nodes
        STATES[0] = v_old[i];

        for (int j = 0; j<ngates; j++){ // Loop over state variables
            STATES[j+1] = w_old[ngates*i+j];
        }

        computeRates(VOI, CONSTANTS, RATES, STATES, ALGEBRAIC);

//  Update gating variables
        for (int j = 0; j<ngates; j++){ // Loop over state variables
            w_new[ngates*i+j] = w_old[ngates*i+j] + dt*RATES[j+1];
        }

//  Save force value
        fdif = finiteDifference(i, v_old, gamma) ;
        dv[i] = RATES[0]*dt + fdif;

    }

//  update v
    for (int i = 1; i<n-1; i++){ // Loop over nodes
        v_new[i] = v_old[i] + dv[i];

    }

//  Apply Neumann BC
    if (rev == 0){ // left to right
        v_new[n-1] = v_new[n-3]; 
        v_new[0] = v_new[2] - source ;    
    } else if (rev == 1){
        v_new[n-1] = v_new[n-3] - source ; 
        v_new[0] = v_new[2];       
    }   

    free(RATES);    free(STATES);    free(ALGEBRAIC);    free(dv);
    

}




double GjnormSS(double Vj, double* Cxparams){
    double gjmin_pos = Cxparams[0];
    double gjmin_neg = Cxparams[1];
    double vj0_pos = Cxparams[2];
    double vj0_neg = Cxparams[3];
    double A_pos = Cxparams[4];
    double A_neg = Cxparams[5];
    double pp = Cxparams[6];
    double d = Cxparams[7];
    
    double gjnorm = 0;
    
    if (Vj >= d){
        gjnorm = (1.- gjmin_pos)/(pp + exp(A_pos*(Vj-vj0_pos))) + gjmin_pos;
                      
    } 
    else{
        gjnorm = (1.- gjmin_neg)/(pp + exp(A_neg*(Vj-vj0_neg))) + gjmin_neg;
    
    }

    return gjnorm;

}

double fexp(double Vj, double VH){
    
    return exp(Vj/(VH*(1+exp(-Vj/VH))));

}

double GjnormIN(double Vj, double* Cxparams){
    double Gjn = Cxparams[8];
    double Gjp = Cxparams[9];
    double VHn = Cxparams[10];
    double VHp = Cxparams[11];
    
    double gjnorm = 0;
    
    if (Vj >= 0){
        gjnorm = Gjp/(fexp(Vj, VHp) + fexp(-Vj, VHp));
                      
    } 
    else{
        gjnorm = Gjn/(fexp(Vj, VHn) + fexp(-Vj, VHn));
    
    }

    return gjnorm;

}


double computeTaug(double Vj, double* dynparams){

    double sigg = dynparams[0];
    double kg = dynparams[1];    
    
    double taug = 0;
    
    taug = 1./(sigg*(exp(kg*fabs(Vj))));
    
    return taug;


}

void NOupdate(double* v, long int* gappos, double* Cxparams, double* gamparams,
                int ndiv, int ngap, int tcond, double* gjnorm, double* gamma){

    int gap;
    double Vj;
    double rd;
    double sig;
    
    double Am = gamparams[0];
    double Cm = gamparams[1];
    double rd0 = gamparams[3];
    double dx = gamparams[4];
    double dt = gamparams[5];
    double factor = gamparams[6];
    
    
//  Iterate only on gap positions
    for (int i = 0; i<ngap; i++){ // Loop over nodes
        gap = gappos[i];
        Vj = v[gap + 1 + (ndiv-1)] - v[gap - (ndiv-1)];
//        Vj = v[gap + 1] - v[gap];
        
        if (tcond == 1){    // SS conductance function
            gjnorm[gap] = GjnormSS(Vj, Cxparams);        
        }
        else{               // Inst. conductance function
            gjnorm[gap] = GjnormIN(Vj, Cxparams);           
        }
//        printf("%d ", gap);
//        printf("\n");
        rd = rd0/(factor*gjnorm[gap]);
        sig = 1./(rd/dx);
        gamma[gap] = (dt*sig)/(Am*Cm*pow(dx, 2));
        }    



}




void NODupdate(double* v, long int* gappos,  double* dynparams, double* Cxparams,
                 double* gamparams, int ndiv, int ngap, double* gjnorm, 
                 double* gss, double* taug, double* gamma){

    int gap;
    double Vj;
    double rd;
    double sig;
    
    double Am = gamparams[0];
    double Cm = gamparams[1];
    double rd0 = gamparams[3];
    double dx = gamparams[4];
    double dt = gamparams[5];
    double factor = gamparams[6];
    
    
//  Iterate only on gap positions
    for (int i = 0; i<ngap; i++){ // Loop over nodes
        gap = gappos[i];
        Vj = v[gap + 1 + (ndiv-1)] - v[gap - (ndiv-1)];
//        Vj = v[gap + 1] - v[gap];
        
        gss[gap] = GjnormSS(Vj, Cxparams);
        taug[gap] = computeTaug(Vj, dynparams);
//        printf("%f ", taug[gap]);
//        printf("\n");
    
        
        gjnorm[gap] = gjnorm[gap] + dt*(gss[gap] - gjnorm[gap])/taug[gap];
        
        rd = rd0/(factor*gjnorm[gap]);
        sig = 1./(rd/dx);
        gamma[gap] = (dt*sig)/(Am*Cm*pow(dx, 2));
        }    



}
