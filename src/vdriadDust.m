classdef Dust < handle
    %DUST class to store the information of dust particles in the
    %simulation
    %   Detailed explanation goes here
    properties (SetAccess=public)
        Mass 
        Diameter
        Radius
        Position 
        Velocity 
        Acceleration
        Charge
        WakeChargePercent
        WakeLength
        WakeNNR
        WakeNNZ
        WakeNNId
    end
    methods (Access=public)
        function obj = Dust(params)
        %Dust is the constructor for the Dust class. 
        obj.GenerateParticles(params);
        %obj.MemCpy('HtoD','all');
        end
        function GenerateParticles(obj,params)
        %GenerateParticles creates the particles based on the properties
        %specified by the params class/strcut
        %   Takes the struct 'params' as an argument
        %   and generates the dust particles accordingly

            % Allocate memory
            xVec = zeros(params.NUM_PARTICLES,1,'single');
            yVec = zeros(params.NUM_PARTICLES,1,'single');
            zVec = zeros(params.NUM_PARTICLES,1,'single');
        
            x = 0;
            y = 0;
            z = 0;
        
            for i = 1:params.NUM_PARTICLES
                notValid = true;

                % Generate random position
                while notValid
                    x   = (rand('single') * 2 - 1) * params.CELL_RADIUS;
                    y   = (rand('single') * 2 - 1) * params.CELL_RADIUS;
                    z   = (rand('single') * 0.5 + 1 - 0.5) * params.CELL_HEIGHT;
                    
                    % Check for particles that are too close
                    isClose = any(((xVec(1:i-1)-x).^2 + (yVec(1:i-1)-y).^2 + (zVec(1:i-1)-z).^2) < (10000e-12));
        
                    % Check for particles outside the cell
                    isOutside = (x^2 + y^2) >= (params.CELL_RADIUS^2);
        
                    notValid = (isClose) || (isOutside);
                end
                
                % Save coordinates
                xVec(i) = x;
                yVec(i) = y;
                zVec(i) = z;
            end
        
            % SAVE PARTICLE POSITIONS
            obj.Position.host.x = xVec;
            obj.Position.host.y = yVec;
            obj.Position.host.z = zVec;

            % GENERATE PARTICLE VELOCITIES
            obj.Velocity.host.x = zeros(params.NUM_PARTICLES,1,'single');
            obj.Velocity.host.y = zeros(params.NUM_PARTICLES,1,'single');
            obj.Velocity.host.z = zeros(params.NUM_PARTICLES,1,'single');
            
            % GENERATE PARTICLE ACCELERATIONS
            obj.Acceleration.host.x = zeros(params.NUM_PARTICLES,1,'single');
            obj.Acceleration.host.y = zeros(params.NUM_PARTICLES,1,'single');
            obj.Acceleration.host.z = zeros(params.NUM_PARTICLES,1,'single');
            
            % GENERATE SIZE, MASS AND CHARGE
            obj.Diameter.host = (randn(params.NUM_PARTICLES,1,'single') * params.DUST_DIAMETER_STD + params.DUST_DIAMETER_MEAN);
            obj.Radius.host   = obj.Diameter.host / 2;
            obj.Mass.host     = params.DUST_DENSITY * (4/3 * pi * obj.Radius.host.^3);
            obj.Charge.host   = round(params.DUST_CHARGE_DENSITY_MEAN * obj.Diameter.host) * params.ELECTRON_CHARGE;
        
            % GENERATE ION WAKE
            obj.WakeChargePercent.host          = params.WAKE_CHARGE_PERCENT * ones(params.NUM_PARTICLES,1,'single');
            obj.WakeLength.host                 = params.WAKE_LENGTH * ones(params.NUM_PARTICLES,1,'single');
            obj.WakeNNR.host                    = 1000 * params.ION_DEBYE * ones(params.NUM_PARTICLES,1,'single');
            obj.WakeNNZ.host                    = 1000 * params.ION_DEBYE * ones(params.NUM_PARTICLES,1,'single');
            obj.WakeNNId.host                   = -1 * ones(params.NUM_PARTICLES,1,'int32');
        
            % SAVE INITIAL CONDITIONS
            obj.Position.initial.x          = xVec;
            obj.Position.initial.y          = yVec;
            obj.Position.initial.z          = zVec;
            obj.Velocity.initial.x          = obj.Velocity.host.x;
            obj.Velocity.initial.y          = obj.Velocity.host.y;
            obj.Velocity.initial.z          = obj.Velocity.host.z;
            obj.Acceleration.initial.x      = obj.Acceleration.host.x;
            obj.Acceleration.initial.y      = obj.Acceleration.host.y;
            obj.Acceleration.initial.z      = obj.Acceleration.host.z;
    end
        function MemCpy(obj,direction,varargin)
            %MemCpy Moves data between the GPU and the host
            copyAll = false;
            if nargin > 3
                error('Error: Invalid number of input arguments')
            elseif nargin == 3
                opt = varargin{1};
                if strcmp(opt,'all')
                    copyAll = true;
                end
            end
        
            if strcmp(direction,'HtoD')
                % COPY POSITIONS
                obj.Position.device.x = gpuArray(obj.Position.host.x);
                obj.Position.device.y = gpuArray(obj.Position.host.y);
                obj.Postioin.device.z = gpuArray(obj.Position.host.z);
        
                if copyAll
                    % COPY VELOCITIES
                    obj.Velocity.device.x = gpuArray(obj.Velocity.host.x);
                    obj.Velocity.device.y = gpuArray(obj.Velocity.host.y);
                    obj.Velocity.device.z = gpuArray(obj.Velocity.host.z);
            
                    % COPY ACCELERATIONS
                    obj.Acceleration.device.x = gpuArray(obj.Acceleration.host.x);
                    obj.Acceleration.device.y = gpuArray(obj.Acceleration.host.y);
                    obj.Acceleration.device.z = gpuArray(obj.Acceleration.host.z);
        
                    % COPY DUST PARAMETERS
                    obj.Diameter.device = gpuArray(obj.Diameter.Host);
                    obj.Radius.device   = gpuArray(obj.Radius.Host);
                    obj.Charge.device   = gpuArray(obj.Charge.Host);
                    obj.Mass.device     = gpuArray(obj.Mass.Host);
        
                    obj.WakeChargePercent.device  = gpuArray(obj.WakeChargePercent.host);
                    obj.WakeLength.device         = gpuArray(obj.WakeLength.host);
                    obj.WakeNNR.device          = gpuArray(obj.WakeNNR.host);
                    obj.WakeNNZ.device          = gpuArray(obj.WakeNNZ.host);
                    obj.WakeNNId.device         = gpuArray(obj.WakeNNId.host);
                end
            elseif strcmp(direction,'DtoH')
                % COPY POSITIONS
                obj.Position.host.x = gather(obj.Position.device.x);
                obj.Position.host.y = gather(obj.Position.device.y);
                obj.Position.host.z = gather(obj.Position.device.z);
                
                if copyAll
                    % COPY VELOCITIES
                    obj.Velocity.host.x = gather(obj.Velocity.device.x);
                    obj.Velocity.host.y = gather(obj.Velocity.device.y);
                    obj.Velocity.host.z = gather(obj.Velocity.device.z);
            
                    % COPY ACCELERATIONS
                    obj.Acceleration.host.x = gather(obj.Acceleration.device.x);
                    obj.Acceleration.host.y = gather(obj.Acceleration.device.y);
                    obj.Acceleration.host.z = gather(obj.Acceleration.device,z);
                end
            end
        end
    end
end