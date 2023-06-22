classdef vdriadDust < handle
    %DUST class to store the information of dust particles in the
    %simulation
    %   Detailed explanation goes here

    % PROPERTIES===========================================================
    properties (SetAccess=public)
        Mass 
        Diameter
        Radius
        Charge

        % Spatial
        Position 
        Velocity 
        Acceleration
        
        % Wake Characteristics
        WakeCharge
        WakeLength
        WakeNNR
        WakeNNZ
        WakeNNId
    end

    % PUBLIC METHODS=======================================================
    methods (Access=public)
        % Class constructor
        function obj = vdriadDust(params)
        % Call the function that will generate the particles
            obj.generateParticles(params);
        end

        % Name:
        % Description:
        function generateParticles(obj,params)
        %GenerateParticles creates the particles based on the properties
        %specified by the params class/strcut
        %   Takes the struct 'params' as an argument
        %   and generates the dust particles accordingly

            % Allocate memory
            N = params.NUM_PARTICLES;
            xVec = zeros(N,1,'single');
            yVec = zeros(N,1,'single');
            zVec = zeros(N,1,'single');
        
            x = 0;
            y = 0;
            z = 0;
        
            % Generate particles randomly in the specified region. Only
            % keep those particles that are far enough from other particles
            d_tolerance_squared = 10000e-12;

            for i = 1:N
                notValid = true;

                % Generate random position
                while notValid
                    x   = (rand('single') * 2 - 1) * params.CELL_RADIUS;
                    y   = (rand('single') * 2 - 1) * params.CELL_RADIUS;
                    z   = (rand('single') * 0.5 + 1 - 0.5) * params.CELL_HEIGHT;
                    
                    % Check for particles that are too close
                    isClose = any(((xVec(1:i-1)-x).^2 + (yVec(1:i-1)-y).^2 + (zVec(1:i-1)-z).^2) < d_tolerance_squared);
        
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
            obj.Velocity.host.x = zeros(N,1,'single');
            obj.Velocity.host.y = zeros(N,1,'single');
            obj.Velocity.host.z = zeros(N,1,'single');
            
            % GENERATE PARTICLE ACCELERATIONS
            obj.Acceleration.host.x = zeros(N,1,'single');
            obj.Acceleration.host.y = zeros(N,1,'single');
            obj.Acceleration.host.z = zeros(N,1,'single');
            
            % GENERATE SIZE, MASS AND CHARGE
            obj.Diameter.host = (randn(N,1,'single') * params.DUST_DIAMETER_STD + params.DUST_DIAMETER_MEAN);
            obj.Radius.host   = obj.Diameter.host / 2;
            obj.Mass.host     = params.DUST_DENSITY * (4/3 * pi * obj.Radius.host.^3);
            obj.Charge.host   = params.DUST_CHARGE * ones(N,1,'single');
        
            % GENERATE ION WAKE
            obj.WakeCharge.host                 = params.WAKE_CHARGE_RATIO * obj.Charge.host;
            obj.WakeLength.host                 = params.WAKE_LENGTH * ones(N,1,'single');
            obj.WakeNNR.host                    = 1000 * params.ION_DEBYE * ones(N,1,'single');
            obj.WakeNNZ.host                    = 1000 * params.ION_DEBYE * ones(N,1,'single');
            obj.WakeNNId.host                   = -1 * ones(N,1,'int32');

            % SAVE INITIAL CONDITIONS
            obj.saveInitialConditions;
        end

        % Name:             saveInitialConditions
        % Description:      Copies the values currently stored in the host
        % array and stores them as initial conditions
        function saveInitialConditions(obj)
            obj.Position.initial.x          = obj.Position.host.x;
            obj.Position.initial.y          = obj.Position.host.y;
            obj.Position.initial.z          = obj.Position.host.z;
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