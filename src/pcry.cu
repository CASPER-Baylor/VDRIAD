#include "pcry.cuh"

__global__ void pcryCalculate_ACC(
                  float* dustPosX, float* dustPosY, float* dustPosZ,
			  	        float* dustVelX, float* dustVelY, float* dustVelZ,
			  	        float* dustAccX, float* dustAccY, float* dustAccZ,
                  float* dustRadius,
                  float* dustMass,
                  float* dustCharge,
                  float* wakeCharge,
                  float* wakeLength,
                  float* wakeDistanceZ,
                  float* wakeDistanceR,
			  	        int* wakeID, 
			  	        float DUST_RADIUS_MEAN,
				          float COULOMB, 
				          float ION_DEBYE,
				          float ELECTRON_DEBYE,  
				          float CUTOFF_M, 
				          float RADIAL_CONFINEMENT_COEFF,
                          float CELL_RADIUS, 
				          float E_FIELD_COEFF, 
				          float CELL_HEIGHT, 
				          float SHEATH_HEIGHT, 
				          float WAKE_CHARGE_PERCENT,
				          float GRAVITY, 
				          float GAS_TEMP,
                  float GAS_PRESSURE,
                  float TIME_STEP,
                  double TIME,
				          int NUM_PARTICLES){
			  
	// VARIABLE DICTIONARY----------------------------------------------------------------------------------------------
  float 	acc;            // Temporarily stores acceleration
  float   epsilon;        // Softening factor
  float   Ez;             // Electric field by the lower electrode

  // ITH PARTICLE
  float   x1;             // Stores the x position of the ith particle
  float   y1;             // Stores the y position of the ith particle
  float   z1;             // Stores the z position of the ith particle
  float   ax1;            // x component of the accelaration of the ith particle
  float   ay1;            // y component of the acceleration of the ith particle
  float   az1;            // z component of the acceleration of the ith particle
  float   mass1;          // mass of the ith particle
  float   charge1;        // charge of the ith particle
  float   radius1;        // radius of the ith particle

  // JTH PARTICLE         !! This part should actually not be here, what wee need to replace is the variables below
  float   x2;             // Stores the x position of the jth particle
  float   y2;             // Stores the y position of the jth particle
  float   z2;             // z position of the jth particle

  // OTHER VARIABLES
  float   dx;             // Distance between the particles in the x direction
  float   dy;             // Distance between the particles in the y direction
  float   dz;             // Distance between the particles in the z direction
  float   r_squared;      // Norm squared
  float   r;              // Eucledian distance between the particles
  float   r_soft;         // Eucledian distance with softening factor
  float   r_min;
  float   z_min;
  float   yourId;
  float   nn_id;

  float 	accX_i, accY_i, accZ_i; 
  float 	posX_i, posY_i, posZ_i;
  float 	charge_i, mass_i;

  float   SIGMA;
  float   BETA;

  // STATE FOR GENERATING RANDOM NUMBER
  curandState_t state;


	// VARIABLES TO BE ALLOCATED IN SHARED MEMORY
	__shared__ float posX_j[BLOCK], posY_j[BLOCK], posZ_j[BLOCK];
	__shared__ float charge_j[BLOCK], wakeCharge_j[BLOCK], wakeLength_j[BLOCK];

  //------------------------------------------------------------------------------------------------------------------
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	if(i < NUM_PARTICLES){              // Making sure we are not out working past the number of particles
    epsilon  = 1e-6;

    // Save positions
    x1  = dustPosX[i];
    y1  = dustPosY[i];
    z1  = dustPosZ[i];

    posX_i 	 = dustPosX[i];
    posY_i 	 = dustPosY[i];
    posZ_i 	 = dustPosZ[i];

    mass1   = dustMass[i];
    charge1 = dustCharge[i];
    radius1 = dustRadius[i];

    // Initialize forces
    accX_i 	 = 0.0f;
    accY_i 	 = 0.0f;
    accZ_i 	 = 0.0f;

    // Other variables
    r_min 	 = CUTOFF_M * ION_DEBYE;
    z_min 	 = 100000.0;
    nn_id 	 = -1;

    wakeDistanceR[i] = 10000.0f;
    wakeDistanceZ[i] = 10000.0f;
    wakeID[i]	 = -1;
		
		// CALCULATING INTERPARTICLE FORCES-----------------------------------------------------------------------------
		for(int j = 0; j < gridDim.x; j++){
			// Save into shared memory
			posX_j[threadIdx.x] 	= dustPosX[threadIdx.x + blockDim.x*j];
			posY_j[threadIdx.x] 	= dustPosY[threadIdx.x + blockDim.x*j];
			posZ_j[threadIdx.x] 	= dustPosZ[threadIdx.x + blockDim.x*j];
			
			charge_j[threadIdx.x] 	= dustCharge[threadIdx.x + blockDim.x*j];
			wakeCharge_j[threadIdx.x] = wakeCharge[threadIdx.x + blockDim.x*j];
			wakeLength_j[threadIdx.x] = wakeLength[threadIdx.x + blockDim.x*j];
			
			__syncthreads();
			
			#pragma unroll 32
            for(int yourSharedId = 0; yourSharedId < blockDim.x; yourSharedId++){
                yourId = yourSharedId + blockDim.x*j;
		    		
                if(i != yourId && yourId < NUM_PARTICLES){
                    dx 		= posX_j[yourSharedId] - posX_i;
                    dy		= posY_j[yourSharedId] - posY_i;
                    dz 		= posZ_j[yourSharedId] - posZ_i;

                    // Norm squared with smoothing factor
                    r_squared  	= dx*dx + dy*dy + dz*dz;
                    r_soft      = sqrt(r_squared + (epsilon * epsilon));
                    r		= sqrt(r_squared);
                    
                    // DUST-DUST YUKAWA FORCE
                    acc  = -COULOMB*charge_j[yourSharedId]*charge_i*(1.0f+r/ION_DEBYE)*exp(-r/ION_DEBYE)/(r_soft*r_soft);
                    acc /= (mass_i);

                    accX_i	+= 0; //acc * (dx/r_soft);
                    accY_i 	+= 0; //acc * (dy/r_soft);
                    accZ_i 	+= 0; //acc * (dz/r_soft);

                    // Finding the nearest neighbor below the current dust grain and within
                    // the specified distance (6 * ION_DEBYE)
                    // We will use this to set the ionWake of the two dusts in question.
                    // This will be done in the move function to remove any race conditions.
                    if(dz < 0.0f){ // If dz is negative you are below me.
                        if(r < r_min){
                            r_min	= r;
                            z_min	= dz;
                            nn_id   = yourId;  // This needs to be the real Id not what is in shared memory.
                        }
                    }
                    


                    // DUST-ION YUKAWA FORCE
                    dz = (posZ_j[yourSharedId] - wakeLength_j[yourSharedId]) - posZ_i;

                    r_squared  	= dx*dx + dy*dy + dz*dz;
                    r_soft	= sqrt(r_squared + (epsilon * epsilon));
                    r  	   	= sqrt(r_squared);

                    acc 	= (COULOMB*charge_j[yourSharedId]*wakeCharge_j[yourSharedId]*charge_i);
                    acc		*=(1.0f + r/ELECTRON_DEBYE)*exp(-r/ELECTRON_DEBYE)/mass_i;
			              acc = 0;	
                    accX_i 	+= acc * (dx/(r_soft*r_soft*r_soft));
                    accY_i 	+= acc * (dy/(r_soft*r_soft*r_soft));
                    accZ_i 	+= acc * (dz/(r_soft*r_soft*r_soft));
				}
			}
		}
		
		wakeDistanceR[i]    		= r_min;	// Saving minimum total distance
		wakeDistanceZ[i]			= z_min;	// Saving minimum y distance
		wakeID[i]			 	    = nn_id; 	// Saving the nearest neighbor's ID

        // CALCULATING EXTERNAL FORCES----------------------------------------------------------------------------------
		if (z1 > SHEATH_HEIGHT){
            Ez = 0;
            }
        else{
            Ez = -2*E_FIELD_COEFF*(z1 - SHEATH_HEIGHT);
        }

		accZ_i += charge1 * Ez / mass1;

		
		// RADIAL CONFINEMENT FORCE
		// r 	= sqrt(x1*x1+y1*y1);
		// if(r != 0){
		//     acc = charge_i*RADIAL_CONFINEMENT_COEFF*pow(r/CELL_RADIUS,12)/mass_i;
		//     accX_i += acc * (posX_i/r);
		//     accY_i += acc * (posY_i/r);
		// }

		// GRAVITATIONAL FORCE
		accZ_i += -GRAVITY;
		
		// DRAG FORCE
        // BETA = 1.44* 4.0 /3.0 * (radius1*radius1) * GAS_PRESSURE / mass1 * sqrt(8.0 * PI * ION_MASS/BOLTZMANN/GAS_TEMP);
        // 
		// accX_i += -BETA * dustVelX[i];
		// accY_i += -BETA * dustVelY[i];
		// accZ_i += -BETA * dustVelZ[i];

        // BROWNIAN MOTION
        // curand_init((time_t)(TIME+i),0,0,&state);
        // SIGMA = sqrt(2.0* BETA * BOLTZMANN * GAS_TEMP/mass1/TIME_STEP);
        // 
        // accX_i += SIGMA * curand_normal(&state);
        // accY_i += SIGMA * curand_normal(&state);
        // accZ_i += SIGMA * curand_normal(&state);

        // LOAD FORCES--------------------------------------------------------------------------------------------------
        // If the dust grain gets too close or passes through the floor. I put it at the top of the sheath, set its
        // force to zero and set its mass, charge and diameter to the base (maybe it was too heavy).
		if(0 < posZ_i){
			dustAccX[i] = accX_i;
			dustAccY[i] = accY_i;
			dustAccZ[i] = accZ_i;
		} else{
			dustPosZ[i] = SHEATH_HEIGHT;

			dustVelX[i] 	= 0.0;
			dustVelY[i] 	= 0.0;
			dustVelZ[i] 	= 0.0;
			
			dustAccX[i]     = 0.0;
			dustAccY[i]     = 0.0;
			dustAccZ[i]     = 0.0;
		}
	}
}


__global__ void pcryCalculate_POS(
                  float* dustPosX, float* dustPosY, float* dustPosZ,
                  float* dustVelX, float* dustVelY, float* dustVelZ,
                  float* dustAccX, float* dustAccY, float* dustAccZ,
                  float DT,
                  float TIME,
                  int 	NUM_PARTICLES){
	// Moving the system forward in time with leap-frog and randomly adjusting the charge on each dust particle.
	curandState state;
	float randomNumber;
	float cutOff, reduction;
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	
	// Note: DustForce.w hold the mass of the dust grain.
	if(i < NUM_PARTICLES){
		if(TIME == 0.0f){
			dustVelX[i] += 0.5f*DT*dustAccX[i];
			dustVelY[i] += 0.5f*DT*dustAccY[i];
			dustVelZ[i] += 0.5f*DT*dustAccZ[i];
		} else {
			dustVelX[i] += DT*dustAccX[i];
			dustVelY[i] += DT*dustAccY[i];
			dustVelZ[i] += DT*dustAccZ[i];
		}

		dustPosX[i] += dustVelX[i]*DT;
		dustPosY[i] += dustVelY[i]*DT;
		dustPosZ[i] += dustVelZ[i]*DT;
	}				
}







