classdef Param < handle
	% Universal Constants
	properties (Access=public,Constant)
		COULOMB         = 8.98e12;
		BOLTZMANN       = 1.38e-23;
		PERMITTIVITY    = 8.85e-12;
        ELECTRON_CHARGE = -1.60e-19;
    end

    % Dynamic Parameters (Updated automatically by the simulation)
    properties (GetAccess=public,SetAccess=private)
        BLOCK_SIZE
		IONIZATION_FRAC
		GAS_DENSITY
        SHEATH_HEIGHT

        ION_DEBYE
        ION_MASS
		ION_DENSITY

		ELECTRON_TEMPERATURE
		ELECTRON_DENSITY
		ELECTRON_DEBYE

		WAKE_CHARGE_RATIO
        WAKE_CHARGE
		WAKE_LENGTH_RATIO
        WAKE_LENGTH

        NUM_PARTICLES
		DUST_DENSITY
		DUST_DIAMETER_MEAN
        DUST_RADIUS_MEAN
        DUST_DIAMETER_STD
		DUST_CHARGE_DENSITY_MEAN
		DUST_CHARGE_DENSITY_STD
        DUST_CHARGE

        CELL_CHARGE
        CELL_RADIUS
		CELL_HEIGHT
		E_FIELD_COEFF
    end

	% Input parameters (Parameters that can be modified by the user)
	properties (Access=public)
        GAS_TEMPERATURE
		GAS_PRESSURE
		GRAVITY
		CELL_POWER
		TIME_STEP
	end

	% Internal Properties
	properties (Access=private)
        FileName
        PathToLut = '../data/LUTData.mat';

        % Parameter ranges
        minPressure     = 0.66;
        maxPressure     = 1.5;
        minPower        = 0.1;
        maxPower        = 20;

        % Functions to interpolate values
        funcElectronDensity
        funcSheathHeight
        funcElectricField
        funcWakeChargeRatio
        funcDustCharge
        funcWakeLengthRatio

        % Internal
        INIT = false;
       
        DRAW_PERIOD
    end

	% Public Methods
	methods (Access=public)
		% Class constructor
		function obj = Param(fileName)
            % Initialize parameter file name
            obj.FileName = fileName;

            % Initialize lookup tables
            obj.readLUT();

            % Read parameters from parameter file
            obj.readParameterFile();
    
            % Calculate dependent parameters
            obj.updateParameters();

            % Update remaining parameters
            obj.TIME_STEP               = 0.0003;
            obj.DRAW_PERIOD             = floor(1/(80 * obj.TIME_STEP));
            obj.DUST_RADIUS_MEAN        = obj.DUST_DIAMETER_MEAN / 2;

            % Initialize 
            obj.INIT = true;
        end
    end

    methods
        % Class Setters
        function set.GAS_TEMPERATURE(obj,val)
            obj.GAS_TEMPERATURE = val;
        end

        function set.GAS_PRESSURE(obj,val)
            if val > obj.maxPressure
                obj.GAS_PRESSURE = obj.maxPressure;
            elseif val < obj.minPressure
                obj.GAS_PRESSURE = obj.minPressure;
            else 
                obj.GAS_PRESSURE = val;

                if obj.INIT
                    obj.updateParameters;
                end
            end
        end

        function set.GRAVITY(obj,val)
            if val > 0
                obj.GRAVITY = val;
            else
                disp('Invalid value for GRAVITY (must be > 0)')
            end
        end

        function set.CELL_POWER(obj,val)
            if val > obj.maxPower
                obj.CELL_POWER = obj.maxPower;
            elseif val < obj.minPower
                obj.CELL_POWER = obj.minPower;
            else
                obj.CELL_POWER = val;

                if obj.INIT
                    obj.updateParameters;
                end
            end
        end

        function set.TIME_STEP(obj,val)
            if (val > 0) && (val < 0.005)
                obj.TIME_STEP = val;
            end
        end
    end

	% Private Methods
	methods (Access=private)
        % Name:             
        % Description:
        function updateParameters(obj)
            % Debye Length anonymous function
            debyeLength = @(T,n) sqrt(obj.PERMITTIVITY * obj.BOLTZMANN * T/(n * (obj.ELECTRON_CHARGE^2)));

            % Calculate neutral gas density
            obj.GAS_DENSITY = obj.GAS_PRESSURE / (obj.BOLTZMANN * obj.GAS_TEMPERATURE);
            
            % Calculate Electron and Ion densities
            obj.ELECTRON_DENSITY = obj.funcElectronDensity(obj.GAS_PRESSURE,obj.CELL_POWER);
            obj.ION_DENSITY = obj.ELECTRON_DENSITY;

            % Calculate Electron and Ion Debye lengths
            obj.ION_DEBYE = debyeLength(obj.GAS_TEMPERATURE,obj.ION_DENSITY);
            obj.ELECTRON_DEBYE = debyeLength(obj.ELECTRON_TEMPERATURE,obj.ELECTRON_DENSITY);

            % Calculate Sheath Height
            obj.SHEATH_HEIGHT = obj.funcSheathHeight(obj.GAS_PRESSURE,obj.CELL_POWER);
            
            % Calculate Dust Charge
            obj.DUST_CHARGE = obj.funcDustCharge(obj.GAS_PRESSURE);

            % Calculate Wake parameters
            obj.WAKE_CHARGE_RATIO = obj.funcWakeChargeRatio(obj.GAS_PRESSURE);
            obj.WAKE_CHARGE = obj.WAKE_CHARGE_RATIO * obj.DUST_CHARGE;

            obj.WAKE_LENGTH_RATIO = obj.funcWakeLengthRatio(obj.GAS_PRESSURE);
            obj.WAKE_LENGTH = obj.WAKE_LENGTH_RATIO * obj.ELECTRON_DEBYE;

            % Calculate Electric Field profile
            obj.E_FIELD_COEFF = obj.funcElectricField(obj.GAS_PRESSURE,obj.CELL_POWER);
        end

        % Name:
        % Description:
        function readParameterFile(obj)
            % Specify read function
            load = @(string) obj.loadParameter(obj.FileName,string);

            obj.BLOCK_SIZE              = load('BLOCK_SIZE');
    
            % GAS
            obj.GAS_TEMPERATURE         = load('GAS_TEMPERATURE');
            obj.GAS_PRESSURE            = load('GAS_PRESSURE');
            obj.ION_MASS                = load('ION_MASS');
            
            % ELECTRON
            obj.ELECTRON_TEMPERATURE    = load('ELECTRON_TEMPERATURE');
    
            obj.NUM_PARTICLES           = load('NUM_PARTICLES');
            obj.DUST_DENSITY            = load('DUST_DENSITY');
            obj.DUST_DIAMETER_MEAN      = load('DUST_DIAMETER_MEAN');

            obj.DUST_DIAMETER_STD       = load('DUST_DIAMETER_STD');
            obj.DUST_CHARGE_DENSITY_MEAN= load('DUST_CHARGE_DENSITY_MEAN');
            obj.DUST_CHARGE_DENSITY_STD = load('DUST_CHARGE_DENSITY_STD');
            
            % CONSTANTS
            obj.GRAVITY                 = load('GRAVITY');
            obj.CELL_CHARGE             = load('CELL_CHARGE');
            obj.CELL_POWER              = load('CELL_POWER');
            obj.CELL_RADIUS             = load('CELL_RADIUS');
            obj.CELL_HEIGHT             = load('CELL_HEIGHT');
        end

        % Name:
        % Description:
        function readLUT(obj)
            % Load array containing LUT data
            load(obj.PathToLut,'LUTArray');

            % Define a function to index the array
            findLUT = @(LUTName) LUTArray{cellfun(@(cell) strcmp(cell.('Name'),LUTName),LUTArray)};
    
            % Load data and function for Electron Density
            LUT = findLUT("LUT01");
            obj.funcElectronDensity = @(pressure,power) interp2(LUT.Pressure,...
                                                                LUT.Power,...
                                                                LUT.ElectronDensity,...
                                                                pressure,...
                                                                power);

            % Load data and function for Sheath Height and Electric field
            LUT = findLUT("LUT02");
            obj.funcSheathHeight  = @(pressure,power) interp2(LUT.Pressure,...
                                                              LUT.Power,...
                                                              LUT.SheathHeight,...
                                                              pressure,...
                                                              power);

            obj.funcElectricField = @(pressure,power) interp2(LUT.Pressure,...
                                                              LUT.Power,...
                                                              LUT.Beta,...
                                                              pressure,...
                                                              power);

            % Load data and function for Wake Charge Ratio, Dust Charge and
            % Wake Length Ratio
            LUT = findLUT("LUT03");
            obj.funcWakeChargeRatio = @(pressure) polyval(LUT.ChargeRatio,pressure);
            obj.funcDustCharge = @(pressure) polyval(LUT.DustCharge,pressure);
            obj.funcWakeLengthRatio = @(pressure) polyval(LUT.LengthRatio,pressure);
        end  
    end

    % Name:
    % Description:
    methods (Static)
        function param = loadParameter(fileName,string)
			isMatch = false;

    		% Open the file
    		fileID = fopen(fileName,'r');

    		% Read line and process if not comment line
    		while (~feof(fileID)) && (~isMatch)
        		str = fgetl(fileID);

        		% Process if not comment line
        		if (~isempty(str)) && (str(1) ~= '!')
        			C = textscan(str,'%s %f32');

        			% Check for name match
        			if strcmp(C{1},string)
            			param = double(C{2});
            			isMatch = true;
        			end
        		end
    		end

    		% Notify if value was not found in parameter file
    		if ~isMatch
        		error('Error: parameter %s not found in %s',name,fileName);
    		end

    		% Close the file
    		fclose(fileID);
        end
    end
end
