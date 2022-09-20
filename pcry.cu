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
				  float DEBYE, 
				  float CUTOFF_M, 
				  float CELL_RADIUS, 
				  float CELL_CHARGE, 
				  float CELL_HEIGHT, 
				  float SHEATH_HEIGHT, 
				  float WAKE_CHARGE_PERCENT,
				  float GRAVITY, 
				  float BETA,
				  int 	NUM_PARTICLES){
			  
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

    // JTH PARTICLE
    float   x2;             // Stores the x position of the jth particle
    float   y2;             // Stores the y position of the jth particle
    float   z2;             // z position of the jth particle
    float   ax2;            // x component of the acceleration of the jth particle
    float   ay2;            // y component of the acceleration of the jth particle
    float   az2;            // z component of the acceleration of the jth particle

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


    float   zi,zi2;
	float 	accX_i, accY_i, accZ_i; 
	float 	posX_i, posY_i, posZ_i;
	float 	charge_i, mass_i;

	// VARIABLES TO BE ALLOCATED IN SHARED MEMORY
	__shared__ float posX_j[BLOCK], posY_j[BLOCK], posZ_j[BLOCK];
	__shared__ float charge_j[BLOCK], wakeCharge_j[BLOCK], wakeLength_j[BLOCK];

    //------------------------------------------------------------------------------------------------------------------
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	if(i < NUM_PARTICLES){              // Making sure we are not out working past the number of particles.
        epsilon  = 2 * dustRadius[i];

		// Save positions
		posX_i 	 = dustPosX[i];
		posY_i 	 = dustPosY[i];
		posZ_i 	 = dustPosZ[i];
		
		// Load other attributes
		charge_i = dustCharge[i];
		mass_i	 = dustMass[i];
		
		// Initialize forces
		accX_i 	 = 0.0f;
		accY_i 	 = 0.0f;
		accZ_i 	 = 0.0f;
		
		// Other variables
		r_min 	 = CUTOFF_M * DEBYE;
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
                    r		    = sqrt(r_squared);
                    
                    // DUST-DUST YUKAWA FORCE
                    acc  = -COULOMB*charge_j[yourSharedId]*charge_i*(1.0f + r/DEBYE)*exp(-r/DEBYE)/r_soft;
                    acc /= (mass_i);


                    accX_i		+= acc * (dx/r);
                    accY_i 		+= acc * (dy/r);
                    accZ_i 		+= acc * (dz/r);

                    // Finding the nearest neighbor below the current dust grain and within
                    // the specified distance (6 * DEBYE)
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

                    r_squared  	= dx*dx + dy*dy + dz*dz + 1.0e-6;
                    r  	   	= sqrt(r_squared);

                    acc 	= (COULOMB*charge_j[yourSharedId]*wakeCharge_j[yourSharedId]*charge_i)/r_squared;
                    acc		*=(1.0f + r/DEBYE)*exp(-r/DEBYE)/mass_i;

                    accX_i 		+= acc * (dx/r);
                    accY_i 		+= acc * (dy/r);
                    accZ_i 		+= acc * (dz/r);
				}
			}
		}
		
		wakeDistanceR[i]    		= r_min;	// Saving minimum total distance
		wakeDistanceZ[i]			= z_min;	// Saving minimum y distance
		wakeID[i]			 	    = nn_id; 	// Saving the nearest neighbor's ID

        // CALCULATING EXTERNAL FORCES----------------------------------------------------------------------------------
        zi = posZ_i;
        zi2 = zi * zi;

		Ez = -8083 + 553373*zi + 2.0e8*zi2 -3.017e10*zi*zi2 + 1.471e12*zi2*zi2 - 2.306e13*zi2*zi2*zi;
		accZ_i += charge_i * Ez / mass_i;
		
		// Calculating radial confinement force
		r  	= sqrt(posX_i*posX_i + posY_i*posY_i);
		if(r != 0){
		    acc = charge_i*CELL_CHARGE*pow(r/CELL_RADIUS,12)/mass_i;
		    accX_i += acc * (posX_i/r);
		    accY_i += acc * (posY_i/r);
		}

		
		// Calculate acceleration due to gravity
		accZ_i += -GRAVITY;
		
		// Calculating drag force
		accX_i += -BETA * dustVelX[i];
		accY_i += -BETA * dustVelY[i];
		accZ_i += -BETA * dustVelZ[i];
		
		if(DUST_RADIUS_MEAN < posZ_i){
			dustAccX[i] = accX_i;
			dustAccY[i] = accY_i;
			dustAccZ[i] = accZ_i;
		} else{ // If the dust grain gets too close or passes through the floor. I put it at the top of the sheath, set its force to zero and set its mass, charge and diameter to the base (maybe it was too heavy).
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
				  float* dustRadius,
				  float* dustMass,
                  float* dustCharge,
                  float* wakeCharge,
				  float* wakeLength,
                  float* wakeDistanceZ,
                  float* wakeDistanceR,
				  int* wakeID,
			  	  float DUST_CHARGE_DENSITY_MEAN,
			  	  float ELECTRON_CHARGE,
				  float CUTOFF_M,  
				  float WAKE_CHARGE_PERCENT, 
				  float WAKE_LENGTH,
				  float DEBYE,
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
        /*
		// Updating my ionwake percent charge and length below the dust if its distance to the its nearest dust grain is within debyeLengthMultiplier*debyeLengths.
		// Also adding the percent charge that the upstream ionWake lost to the down stream ionWake.
		// Need to do this before you update the positions or you will get a miss read on dy.
		if(wakeDistanceZ[i] < 0.0){ // It was initialized to 100000.0 and if no dust grain is close enough it will stay -100000.0.
			cutOff = CUTOFF_M*DEBYE;
			// This is a quadratic function that goes from 1 to 0 as the dust-dust distance goes from debyeLengthMultiplier*debyeLength to zero.
			// Using a second order because as the bottom dust ets close to the top dust it will be eating up a ring (second order) of ions that would have added to the ionwake.
			// It will be used to decrease the top dust's ionwake and give this lose to the bottom dust's ionwake.
			reduction = (1.0f - wakeDistanceR[i]*wakeDistanceR[i]/(cutOff*cutOff))*(wakeDistanceZ[i] /
				     wakeDistanceR[i])*(wakeDistanceZ[i]/wakeDistanceR[i]);
				     
			
			wakeCharge[i]		    = WAKE_CHARGE_PERCENT - WAKE_CHARGE_PERCENT*reduction;	// Decreasing the top charge
			wakeCharge[wakeID[i]]	= WAKE_CHARGE_PERCENT + WAKE_CHARGE_PERCENT*reduction;	// Increasing the bottom charge
			
			// This is a linear that goes from 1 to 0 as the dust-dust distance goes from debyeLengthMultiplier*debyeLength to zero.
			// Using a first order because the as the bottom dust moves up linearly it will displace a ring of ions that would have added to the ionwake.
			// It will be used to decrease the top dust's ionwake length below the top dust.
			//reduction = ionWake[id].z/cutOff;
			reduction = (1.0f - wakeDistanceR[i]/(cutOff))*(wakeDistanceZ[i]/wakeDistanceR[i])*(wakeDistanceZ[i]/wakeDistanceR[i]);
			wakeLength[i] = WAKE_LENGTH - WAKE_LENGTH*reduction;
		} else{
			// If for some reason the ionwake didn't get turned back on it is reset here.
			wakeCharge[i] = WAKE_CHARGE_PERCENT;
			wakeLength[i] = WAKE_LENGTH;
		}	*/
		
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

        /*
		// Randomly perturbating the dust electron count. 
		// This gets a little involved. I first get a standard normal distributed number (Mean 0 StDev 1).
		// Then I set its StDev to the number of electrons that fluctuate per unit dust diameter for this dust grain size.
		// Then I set the mean to how much above or below the base electron per unit dust size.
		// ie. if it has more than it should it has a higher prob of losing and vice versa if it has less than it should.
		// This is just what I came up with and it could be wrong but below is how I did this.
		// dustPos.w carries the charge and dustVel.w carries the diameter.
		
		// Initailizing the cudarand function.
		curand_init(clock64(), i, 0, &state);
		// This gets a random number with mean 0.0 and stDev 1.0;.
		randomNumber = curand_normal(&state);
		// This sets the electron fluctuation for this sized dust grain and makes it the stDev.
		randomNumber *= DUST_CHARGE_DENSITY_MEAN*(2 * dustRadius[i]);
		
		// This has a mean of zero which would just create a random walk. I don't think this is what you want.
		// Dust grains with more electrons than they should have should in general loose electrons 
		// and those with less than they should should in general gain more electrons.
		// We will accomplish this by setting the mean to be the oposite of how much above or below 
		// the base amount you are at this time.
		// This works out to be base number - present number
		randomNumber += DUST_CHARGE_DENSITY_MEAN*(2*dustRadius[i]) - dustCharge[i]/ELECTRON_CHARGE;
		
		// Now add/subtract this number of electron to the existing charge.
    		dustCharge[i] += randomNumber*ELECTRON_CHARGE;
	   
	    	// If the amount of charge ends up being negative which probablistically it could, set it to zero
	    	if(dustCharge[i] < 0.0) dustCharge[i] = 0.0;*/
	}				
}







