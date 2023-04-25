classdef Param
	% Universal Constants
	properties (Access=public,Constant)
		COULOMB         = 8.98e12;
		BOLTZMANN       = 1.38e-23;
		PERMITTIVITY    = 8.85e-12;
	end

	% Simulation Parameters
	properties (Access=public)
		BLOCK_SIZE

		% Gas parameters
		GAS_TEMPERATURE
		GAS_PRESSURE
		ION_MASS
		IONIZATION_FRAC
		ION_DEBYE
		GAS_DENSITY
		ION_DENSITY
		ELECTRON_TEMPERATURE
		ELECTRON_CHARGE
		ELECTRON_DENSITY
		ELECTRON_DEBYE

		SHEATH_HEIGHT
		WAKE_CHARGE_PERCENT
		WAKE_LENGTH
		CUTOFF_MULTIPLIER

		NUM_PARTICLES
		DUST_DENSITY
		DUST_DIAMETER_MEAN
		DUST_RADIUS_MEAN
		DUST_DIAMETER_STD
		DUST_CHARGE_DENSITY_MEAN
		DUST_CHARGE_DENSITY_STD

		GRAVITY

		CELL_CHARGE
		CELL_POWER
		CELL_RADIUS
		CELL_HEIGHT
		E_FIELD

		TIME_STEP

		DRAW_PERIOD
	end

	% Internal Properties
	properties (Access=private)
        FileName
    end
    
	% Public Methods
	methods (Access=public)
		% Class constructor
		function obj = Param(fileName)
            % Store the parameter file name
            obj.FileName = fileName;
            load = @(string) loadParameter(obj,string);

            obj.BLOCK_SIZE              = load('BLOCK_SIZE');
    
            % GAS
            obj.GAS_TEMPERATURE         = load('GAS_TEMPERATURE');
            obj.GAS_PRESSURE            = load('GAS_PRESSURE');
            obj.ION_MASS                = load('ION_MASS');
            obj.IONIZATION_FRAC         = load('IONIZATION_FRAC');
            obj.ION_DEBYE               = nan;
            obj.ELECTRON_DEBYE          = nan;
            obj.GAS_DENSITY             = nan;
            obj.ION_DENSITY             = nan;
            
            % ELECTRON
            obj.ELECTRON_TEMPERATURE    = load('ELECTRON_TEMPERATURE');
            obj.ELECTRON_CHARGE         = load('ELECTRON_CHARGE');
            obj.ELECTRON_DENSITY        = nan;
            obj.ELECTRON_DEBYE          = nan;
            
            obj.SHEATH_HEIGHT           = load('SHEATH_HEIGHT');
            obj.WAKE_CHARGE_PERCENT     = load('WAKE_CHARGE_PERCENT');
            obj.WAKE_LENGTH             = load('WAKE_LENGTH');
            obj.CUTOFF_MULTIPLIER       = load('CUTOFF_MULTIPLIER');
    
            obj.NUM_PARTICLES           = load('NUM_PARTICLES');
            obj.DUST_DENSITY            = load('DUST_DENSITY');
            obj.DUST_DIAMETER_MEAN      = load('DUST_DIAMETER_MEAN');
            obj.DUST_RADIUS_MEAN        = nan;
            obj.DUST_DIAMETER_STD       = load('DUST_DIAMETER_STD');
            obj.DUST_CHARGE_DENSITY_MEAN= load('DUST_CHARGE_DENSITY_MEAN');
            obj.DUST_CHARGE_DENSITY_STD = load('DUST_CHARGE_DENSITY_STD');
            
            % CONSTANTS
            obj.GRAVITY                 = load('GRAVITY');
    
            obj.CELL_CHARGE             = load('CELL_CHARGE');
            obj.CELL_POWER              = load('CELL_POWER');
            obj.CELL_RADIUS             = load('CELL_RADIUS');
            obj.CELL_HEIGHT             = load('CELL_HEIGHT');
            obj.E_FIELD                 = load('E_0');
    
            % TIMING
            obj.TIME_STEP               = 0.0003;
    
            % PARAMETERS THAT MUST BE CALCULATED ONCE
            % Draw rate and print rate
            obj.DRAW_PERIOD             = floor(1/(80 * obj.TIME_STEP));
		end
	end

	% Private Methods
	methods (Access=private)
	    function param = loadParameter(obj,string)
            fileName = obj.FileName;
			match = false;

    		% Open the file
    		fileID = fopen(fileName,'r');

    		% Read line and process if not comment line
    		while (~feof(fileID)) && (~match)
        		str = fgetl(fileID);

        		% Process if not comment line
        		if (~isempty(str)) && (str(1) ~= '!')
        			C = textscan(str,'%s %f32');

        			% Check for name match
        			if strcmp(C{1},string)
            			param = double(C{2});
            			match = true;
        			end
        		end
    		end

    		% Notify if value was not found in parameter file
    		if ~match
        		error('Error: parameter %s not found in %s',name,fileName);
    		end

    		% Close the file
    		fclose(fileID);
        end
	end
end
