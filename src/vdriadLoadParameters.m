% File:             vdriadLoadParameters.m
% Author:           Jorge Augusto Martinez-Ortiz
% Date Created:     02.06.2023

function vdriadLoadParameters(app)
%LoadParameters Loads the simulation parameters from the specified
%parameter file. 
%   Detailed explanation goes here

        % PARAMETERS TO BE LOADED FROM THE PARAMETER FILE
        fileParam = app.fileParam;

        app.params.BLOCK_SIZE              = loadParam('BLOCK_SIZE',fileParam);

        % GAS
        app.params.GAS_TEMPERATURE         = loadParam('GAS_TEMPERATURE',fileParam);
        app.params.GAS_PRESSURE            = loadParam('GAS_PRESSURE',fileParam);
        app.params.ION_MASS                = loadParam('ION_MASS',fileParam);
        app.params.IONIZATION_FRAC         = loadParam('IONIZATION_FRAC',fileParam);
        app.params.ION_DEBYE               = nan;
        app.params.ELECTRON_DEBYE          = nan;
        app.params.GAS_DENSITY             = nan;
        app.params.ION_DENSITY             = nan;
        

        % ELECTRON
        app.params.ELECTRON_TEMPERATURE    = loadParam('ELECTRON_TEMPERATURE',fileParam);
        app.params.ELECTRON_CHARGE         = loadParam('ELECTRON_CHARGE',fileParam);
        app.params.ELECTRON_DENSITY        = nan;
        app.params.ELECTRON_DEBYE          = nan;
        
        app.params.SHEATH_HEIGHT           = loadParam('SHEATH_HEIGHT',fileParam);
        app.params.WAKE_CHARGE_PERCENT     = loadParam('WAKE_CHARGE_PERCENT',fileParam);
        app.params.WAKE_LENGTH             = loadParam('WAKE_LENGTH',fileParam);
        app.params.CUTOFF_MULTIPLIER       = loadParam('CUTOFF_MULTIPLIER',fileParam);

        app.params.NUM_PARTICLES           = loadParam('NUM_PARTICLES',fileParam);
        app.params.DUST_DENSITY            = loadParam('DUST_DENSITY',fileParam);
        app.params.DUST_DIAMETER_MEAN      = loadParam('DUST_DIAMETER_MEAN',fileParam);
        app.params.DUST_RADIUS_MEAN        = nan;
        app.params.DUST_DIAMETER_STD       = loadParam('DUST_DIAMETER_STD',fileParam);
        app.params.DUST_CHARGE_DENSITY_MEAN= loadParam('DUST_CHARGE_DENSITY_MEAN',fileParam);
        app.params.DUST_CHARGE_DENSITY_STD = loadParam('DUST_CHARGE_DENSITY_STD',fileParam);
        
        % CONSTANTS
        app.params.COULOMB                 = loadParam('COULOMB',fileParam);
        app.params.BOLTZMANN               = loadParam('BOLTZMANN',fileParam);
        app.params.GRAVITY                 = loadParam('GRAVITY',fileParam);
        app.params.PERMITTIVITY            = loadParam('PERMITTIVITY',fileParam);

        app.params.CELL_CHARGE             = loadParam('CELL_CHARGE',fileParam);
        app.params.CELL_POWER              = loadParam('CELL_POWER',fileParam);
        app.params.CELL_RADIUS             = loadParam('CELL_RADIUS',fileParam);
        app.params.CELL_HEIGHT             = loadParam('CELL_HEIGHT',fileParam);
        app.params.E_0                     = loadParam('E_0',fileParam);

        % TIMING
        app.params.TIME_STEP               = 0.0003;

        % PARAMETERS THAT MUST BE CALCULATED ONCE
        % Draw rate and print rate
        app.drawPeriod                     = floor(1/(app.drawRate * app.params.TIME_STEP));
end

function param = loadParam(name,filename)
            match = false;

            % Open the file
            fileID = fopen(filename,'r');

            % Read line and process if not comment line
            while (~feof(fileID)) && (~match)
                str = fgetl(fileID);

                % Process if not comment line
                if (~isempty(str)) && (str(1) ~= '!')
                    C = textscan(str,'%s %f32');  

                    % Check for name match
                    if strcmp(C{1},name)
                        param = double(C{2});
                        match = true;
                    end
                end
            end

            % Notify if value was not found in parameter file
            if ~match
                error('Error: parameter %s not found in %s',name,filename);
            end

            % Close the file
            fclose(fileID);
        end