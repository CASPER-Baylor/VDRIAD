% File:             vdriadLoadParameters.m
% Author:           Jorge Augusto Martinez-Ortiz
% Date Created:     02.06.2023

function vdriadLoadParameters(app)
%vdriadLoadParameters Loads the simulation parameters
%   Detailed explanation goes here
        app.params.BLOCK_SIZE              = 32;

        % GAS
        app.params.GAS_TEMPERATURE         = 288.15;
        app.params.GAS_PRESSURE            = 10;
        app.params.ION_MASS                = 6.68e-26;
        app.params.IONIZATION_FRAC         = 1/1e6;
        app.params.ION_DEBYE               
        app.params.GAS_DENSITY             
        app.params.ION_DENSITY             
        

        % ELECTRON
        app.params.ELECTRON_TEMPERATURE    = 28815;
        app.params.ELECTRON_CHARGE         = -1.60e-19;
        app.params.ELECTRON_DENSITY
        app.params.ELECTRON_DEBYE       
        
        app.params.SHEATH_HEIGHT           = 10.6e-3;
        app.params.WAKE_CHARGE_PERCENT     = 0.3;
        app.params.WAKE_LENGTH             = 45e-6;
        app.params.CUTOFF_MULTIPLIER       = 6;

        app.params.NUM_PARTICLES           = 100;
        app.params.DUST_DENSITY            = 1.51 * 1e3;
        app.params.DUST_DIAMETER_MEAN      = 8.89e-6;
        app.params.DUST_RADIUS_MEAN
        app.params.DUST_DIAMETER_STD       = 0.01e-6;
        app.params.DUST_CHARGE_DENSITY_MEAN= 1000 * 1e6;
        app.params.DUST_CHARGE_DENSITY_STD = 20 * 1e6;
        
        % CONSTANTS
        app.params.COULOMB                 = 8.98e12;
        app.params.BOLTZMANN               = 1.38e-23;
        app.params.GRAVITY                 = 9.8;
        app.params.PERMITTIVITY            = 8.85418782e-12;

        app.params.CELL_CHARGE             = 1.6e2;
        app.params.CELL_RADIUS             = 0.6e-2;
        app.params.CELL_HEIGHT             = 1.2e-2;
        app.params.E_0                     = 3350;
end