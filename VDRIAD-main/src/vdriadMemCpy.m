function dust = vdriadMemCpy(dust,direction,varargin)
%UNTITLED4 Summary of this function goes here

    all = false;

    if nargin > 3
        error('Error: Invalid number of input arguments')
    elseif nargin == 3
        opt = varargin{1};
        if strcmp(opt,'all')
            all = true;
        end
    end


    if strcmp(direction,'HtoD')
        % Copy Positions
        dust.d_x            = gpuArray(dust.h_x);
        dust.d_y            = gpuArray(dust.h_y);
        dust.d_z            = gpuArray(dust.h_z);

        if all
            % Copy Velocities
            dust.d_vx           = gpuArray(dust.h_vx);
            dust.d_vy           = gpuArray(dust.h_vy);
            dust.d_vz           = gpuArray(dust.h_vz);
    
            % Copy Accelerations
            dust.d_ax           = gpuArray(dust.h_ax);
            dust.d_ay           = gpuArray(dust.h_ay);
            dust.d_az           = gpuArray(dust.h_az);

            % Copy 
            dust.d_diameter       = gpuArray(dust.h_diameter);
            dust.d_radius         = gpuArray(dust.h_radius);
            dust.d_charge         = gpuArray(dust.h_charge);
            dust.d_mass           = gpuArray(dust.h_mass);

            dust.d_wake_charge         = gpuArray(dust.h_wake_charge);
            dust.d_wake_length         = gpuArray(dust.h_wake_length);
            dust.d_wake_nn_r           = gpuArray(dust.h_wake_nn_r);
            dust.d_wake_nn_z           = gpuArray(dust.h_wake_nn_z);
            dust.d_wake_nn_id          = gpuArray(dust.h_wake_nn_id);
        end
    elseif strcmp(direction,'DtoH')
        % Copy Postions
        dust.h_x            = gather(dust.d_x);
        dust.h_y            = gather(dust.d_y);
        dust.h_z            = gather(dust.d_z);
        
        if all
            % Copy Velocities
            dust.h_vx           = gather(dust.d_vx);
            dust.h_vy           = gather(dust.d_vy);
            dust.h_vz           = gather(dust.d_vz);
    
            % Copy Accelerations
            dust.h_ax           = gather(dust.d_ax);
            dust.h_ay           = gather(dust.d_ay);
            dust.h_az           = gather(dust.d_az);
        end
    end
end