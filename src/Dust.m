classdef Dust < Handle
    %DUST class to store the information of dust particles in the
    %simulation
    %   Detailed explanation goes here
    properties (SetAccess = public)
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

    methods
        function obj = Dust(params)
        %Dust is the constructor for the Dust class. 
        GenerateParticles(params);
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
            obj.Mass.host     = params.DUST_DENSITY * (4/3 * pi * radius.^3);
            obj.Charge.host   = round(params.DUST_CHARGE_DENSITY_MEAN * diameter) * params.ELECTRON_CHARGE;
        
            % GENERATE ION WAKE
            obj.WakeChargePercent          = params.WAKE_CHARGE_PERCENT * ones(params.NUM_PARTICLES,1,'single');
            obj.WakeLength                 = params.WAKE_LENGTH * ones(params.NUM_PARTICLES,1,'single');
            obj.WakeNNR                    = 1000 * params.ION_DEBYE * ones(params.NUM_PARTICLES,1,'single');
            obj.WakeNNZ                    = 1000 * params.ION_DEBYE * ones(params.NUM_PARTICLES,1,'single');
            obj.WakeNNId                   = -1 * ones(params.NUM_PARTICLES,1,'int32');
        
            % SAVE INITIAL CONDITIONS
            obj.Position.Initial.x          = xVec;
            obj.Position.Initial.y          = yVec;
            obj.Position.Initial.z          = zVec;
            obj.Velocity.Initial.x          = obj.Velocity.host.x;
            obj.Velocity.Initial.y          = obj.Velocity.host.y;
            obj.Velocity.Initial.z          = obj.Velocity.host.z;
            obj.Acceleration.Initial.x      = obj.Acceleration.host.x;
            obj.Acceleration.Initial.y      = obj.Acceleration.host.y;
            obj.Acceleration.Initial.z      = obj.Acceleration.host.z;
    end

        function outputArg = method1(obj,inputArg)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            outputArg = obj.Property1 + inputArg;
        end
    end
end