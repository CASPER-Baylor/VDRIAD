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
        %specified by the Param class
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
            obj.Position.Host.x = xVec;
            obj.Position.Host.y = yVec;
            obj.Position.Host.z = zVec;

            % GENERATE PARTICLE VELOCITIES
            obj.Velocity.Host.x = zeros(N,1,'single');
            obj.Velocity.Host.y = zeros(N,1,'single');
            obj.Velocity.Host.z = zeros(N,1,'single');
            
            % GENERATE PARTICLE ACCELERATIONS
            obj.Acceleration.Host.x = zeros(N,1,'single');
            obj.Acceleration.Host.y = zeros(N,1,'single');
            obj.Acceleration.Host.z = zeros(N,1,'single');
            
            % GENERATE SIZE, MASS AND CHARGE
            obj.Diameter.Host = (randn(N,1,'single') * params.DUST_DIAMETER_STD + params.DUST_DIAMETER_MEAN);
            obj.Radius.Host   = obj.Diameter.Host / 2;
            obj.Mass.Host     = params.DUST_DENSITY * (4/3 * pi * obj.Radius.Host.^3);
            obj.Charge.Host   = params.DUST_CHARGE * ones(N,1,'single');
        
            % GENERATE ION WAKE
            obj.WakeCharge.Host                 = params.WAKE_CHARGE_RATIO * obj.Charge.Host;
            obj.WakeLength.Host                 = params.WAKE_LENGTH * ones(N,1,'single');
            obj.WakeNNR.Host                    = 1000 * params.ION_DEBYE * ones(N,1,'single');
            obj.WakeNNZ.Host                    = 1000 * params.ION_DEBYE * ones(N,1,'single');
            obj.WakeNNId.Host                   = -1 * ones(N,1,'int32');

            % SAVE INITIAL CONDITIONS
            obj.saveInitialConditions;
        end

        % Name:             saveInitialConditions
        % Description:      Copies the values currently stored in the Host
        %                   array and stores them as initial conditions
        function saveInitialConditions(obj)
            obj.Position.Initial.x          = obj.Position.Host.x;
            obj.Position.Initial.y          = obj.Position.Host.y;
            obj.Position.Initial.z          = obj.Position.Host.z;
            obj.Velocity.Initial.x          = obj.Velocity.Host.x;
            obj.Velocity.Initial.y          = obj.Velocity.Host.y;
            obj.Velocity.Initial.z          = obj.Velocity.Host.z;
            obj.Acceleration.Initial.x      = obj.Acceleration.Host.x;
            obj.Acceleration.Initial.y      = obj.Acceleration.Host.y;
            obj.Acceleration.Initial.z      = obj.Acceleration.Host.z;
        end

        % Name:             loadInitialConditions
        % Description:
        function loadInitialConditions(obj)
            obj.Position.Host.x             = obj.Position.Initial.x;
            obj.Position.Host.y             = obj.Position.Initial.y;
            obj.Position.Host.z             = obj.Position.Initial.z;
            obj.Velocity.Host.x             = obj.Velocity.Initial.x;
            obj.Velocity.Host.y             = obj.Velocity.Initial.y;
            obj.Velocity.Host.z             = obj.Velocity.Initial.z;
            obj.Acceleration.Host.x         = obj.Acceleration.Initial.x;
            obj.Acceleration.Host.y         = obj.Acceleration.Initial.y;
            obj.Acceleration.Host.z         = obj.Acceleration.Initial.z;      
        end

        % Name:             memoryCopy        
        % Description:      Copies the dust array data back and forth
        %                   between the host (CPU) and memory on the device
        %                   (GPU)
        function memoryCopy(obj,direction,varargin)
            %MemCpy Moves data between the GPU and the Host
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
                obj.Position.Device.x = gpuArray(obj.Position.Host.x);
                obj.Position.Device.y = gpuArray(obj.Position.Host.y);
                obj.Postioin.Device.z = gpuArray(obj.Position.Host.z);
        
                if copyAll
                    % COPY VELOCITIES
                    obj.Velocity.Device.x = gpuArray(obj.Velocity.Host.x);
                    obj.Velocity.Device.y = gpuArray(obj.Velocity.Host.y);
                    obj.Velocity.Device.z = gpuArray(obj.Velocity.Host.z);
            
                    % COPY ACCELERATIONS
                    obj.Acceleration.Device.x = gpuArray(obj.Acceleration.Host.x);
                    obj.Acceleration.Device.y = gpuArray(obj.Acceleration.Host.y);
                    obj.Acceleration.Device.z = gpuArray(obj.Acceleration.Host.z);
        
                    % COPY DUST PARAMETERS
                    obj.Diameter.Device = gpuArray(obj.Diameter.Host);
                    obj.Radius.Device   = gpuArray(obj.Radius.Host);
                    obj.Charge.Device   = gpuArray(obj.Charge.Host);
                    obj.Mass.Device     = gpuArray(obj.Mass.Host);
        
                    obj.WakeChargePercent.Device  = gpuArray(obj.WakeChargePercent.Host);
                    obj.WakeLength.Device         = gpuArray(obj.WakeLength.Host);
                    obj.WakeNNR.Device          = gpuArray(obj.WakeNNR.Host);
                    obj.WakeNNZ.Device          = gpuArray(obj.WakeNNZ.Host);
                    obj.WakeNNId.Device         = gpuArray(obj.WakeNNId.Host);
                end

            elseif strcmp(direction,'DtoH')
                % COPY POSITIONS
                obj.Position.Host.x = gather(obj.Position.Device.x);
                obj.Position.Host.y = gather(obj.Position.Device.y);
                obj.Position.Host.z = gather(obj.Position.Device.z);
                
                if copyAll
                    % COPY VELOCITIES
                    obj.Velocity.Host.x = gather(obj.Velocity.Device.x);
                    obj.Velocity.Host.y = gather(obj.Velocity.Device.y);
                    obj.Velocity.Host.z = gather(obj.Velocity.Device.z);
            
                    % COPY ACCELERATIONS
                    obj.Acceleration.Host.x = gather(obj.Acceleration.Device.x);
                    obj.Acceleration.Host.y = gather(obj.Acceleration.Device.y);
                    obj.Acceleration.Host.z = gather(obj.Acceleration.Device,z);
                end
            end
        end
    end
end