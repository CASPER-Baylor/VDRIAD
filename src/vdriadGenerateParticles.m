function [dust,params] = vdriadGenerateParticles(dust,params)
%GenerateParticles Creates all the particles with their respective
%properties
%   Detailed explanation goes here

    % GENERATE PARTICLE POSITIONS
    % Allocate memory
    xx = zeros(params.NUM_PARTICLES,1,'single');
    yy = zeros(params.NUM_PARTICLES,1,'single');
    zz = zeros(params.NUM_PARTICLES,1,'single');

    x = 0;
    y = 0;
    z = 0;

    for i = 1:params.NUM_PARTICLES
        flag = true;

        % Generate random cartesian coordinates
        while (flag == true)
            x   = (rand('single') * 2 - 1) * params.CELL_RADIUS;
            y   = (rand('single') * 2 - 1) * params.CELL_RADIUS;
            z   = (rand('single') * 0.5 + 1 - 0.5) * params.CELL_HEIGHT;
            
            % Check for particles that are too close
            f1 = any(((xx(1:i-1)-x).^2 + (yy(1:i-1)-y).^2 + (zz(1:i-1)-z).^2) < (10000e-12));

            % Check for particles outside the cell
            f2 = (x^2 + y^2) >= (params.CELL_RADIUS^2);

            flag = (f1) || (f2);
        end
        
        % Save coordinates
        xx(i) = x;
        yy(i) = y;
        zz(i) = z;
    end

    % GENERATE PARTICLE VELOCITIES
    vx = zeros(params.NUM_PARTICLES,1,'single');
    vy = zeros(params.NUM_PARTICLES,1,'single');
    vz = zeros(params.NUM_PARTICLES,1,'single');
    
    % GENERATE PARTICLE ACCELERATIONS
    ax = zeros(params.NUM_PARTICLES,1,'single');
    ay = zeros(params.NUM_PARTICLES,1,'single');
    az = zeros(params.NUM_PARTICLES,1,'single');
    
    % GENERATE SIZE, MASS AND CHARGE
    diameter = (randn(params.NUM_PARTICLES,1,'single') * params.DUST_DIAMETER_STD + params.DUST_DIAMETER_MEAN);
    radius   = diameter / 2;
    mass     = params.DUST_DENSITY * (4/3 * pi * radius.^3);
    charge   = round(params.DUST_CHARGE_DENSITY_MEAN * diameter) * params.ELECTRON_CHARGE;

    % GENERATE ION WAKE
    wake_charge_p           = params.WAKE_CHARGE_PERCENT * ones(params.NUM_PARTICLES,1,'single');
    wake_length             = params.WAKE_LENGTH * ones(params.NUM_PARTICLES,1,'single');
    wake_nn_r               = 1000 * params.ION_DEBYE * ones(params.NUM_PARTICLES,1,'single');
    wake_nn_z               = 1000 * params.ION_DEBYE * ones(params.NUM_PARTICLES,1,'single');
    wake_nn_id              = -1 * ones(params.NUM_PARTICLES,1,'int32');

    % SAVE TO HOST
    dust.h_x         = xx;
    dust.h_y         = yy;
    dust.h_z         = zz;
    dust.h_vx        = vx;
    dust.h_vy        = vy;
    dust.h_vz        = vz;
    dust.h_ax        = ax;
    dust.h_ay        = ay;
    dust.h_az        = az;

    dust.h_diameter       = diameter;
    dust.h_radius         = radius;
    dust.h_charge         = charge;
    dust.h_mass           = mass;

    dust.h_wake_charge         = wake_charge_p;
    dust.h_wake_length         = wake_length;
    dust.h_wake_nn_r           = wake_nn_r;
    dust.h_wake_nn_z           = wake_nn_z;
    dust.h_wake_nn_id          = wake_nn_id;

    params.DUST_RADIUS_MEAN = params.DUST_DIAMETER_MEAN / 2;

    % SAVE INITIAL CONDITIONS
    dust.x0          = xx;
    dust.y0          = yy;
    dust.z0          = zz;
    dust.vx0         = vx;
    dust.vy0         = vy;
    dust.vz0         = vz;
    dust.ax0         = ax;
    dust.ay0         = ay;
    dust.az0         = az;

    % Copy all the dust data to the device
    dust = vdriadMemCpy(dust,'HtoD','all');
end