function vdriadUpdateParams(app)
%UPDATEPARAMS is used after a callback function modifies one of the
%interactive simulation parameters such as PRESSURE, POWER, or GRAVITY.
%   Detailed explanation goes here
    params = app.params;

    % CALCULATE NEUTRAL GAS DENSITY
    params.GAS_DENSITY = params.GAS_PRESSURE / (params.BOLTZMANN * params.GAS_TEMPERATURE);
    
    % CALCULATE ION AND ELECTRON DENSITIES
    %params.ELECTRON_DENSITY = params.GAS_DENSITY * params.IONIZATION_FRAC;
    params.ELECTRON_DENSITY = app.LUTS.LUTDensity.LookUp(params.GAS_PRESSURE,params.CELL_POWER) *1e6;
    %params.ION_DENSITY = params.GAS_DENSITY * params.IONIZATION_FRAC;
    params.ION_DENSITY = params.ELECTRON_DENSITY;

    % CALCULATE ION AND ELECTRON DEBYE LENGTHS
    params.ION_DEBYE = sqrt(params.PERMITTIVITY * params.BOLTZMANN * params.GAS_TEMPERATURE/...
                            (params.ION_DENSITY * params.ELECTRON_CHARGE * params.ELECTRON_CHARGE));

    params.ELECTRON_DEBYE = sqrt(params.PERMITTIVITY * params.BOLTZMANN * params.ELECTRON_TEMPERATURE/...
                            (params.ELECTRON_DENSITY * params.ELECTRON_CHARGE * params.ELECTRON_CHARGE));

    % UPDATE TO DISPLAY
    app.UITable2.Data.Value = [params.ION_DEBYE;...
                               params.ELECTRON_DEBYE;...
                               params.ELECTRON_DENSITY]; 
    % SAVE CHANGES
    app.params = params;
end