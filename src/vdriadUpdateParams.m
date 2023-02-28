function vdriadUpdateParams(app)
%UPDATEPARAMS is used after a callback function modifies one of the
%interactive simulation parameters such as PRESSURE, POWER, or GRAVITY.
%   Detailed explanation goes here
    params = app.params;

    % CALCULATE NEUTRAL GAS DENSITY
    params.GAS_DENSITY = params.GAS_PRESSURE / (params.BOLTZMANN * params.GAS_TEMPERATURE);
    
    % CALCULATE ION AND ELECTRON DENSITIES
    params.ELECTRON_DENSITY = params.GAS_DENSITY * params.IONIZATION_FRAC;
    params.ION_DENSITY = params.GAS_DENSITY * params.IONIZATION_FRAC;

    % CALCULATE ION AND ELECTRON DEBYE LENGTHS
    params.ION_DEBYE = sqrt(params.PERMITTIVITY * params.BOLTZMANN * params.GAS_TEMPERATURE/...
                            (params.ION_DENSITY * params.ELECTRON_CHARGE * params.ELECTRON_CHARGE));

    params.ELECTRON_DEBYE = sqrt(params.PERMITTIVITY * params.BOLTZMANN * params.ELECTRON_TEMPERATURE/...
                            (params.ELECTRON_DENSITY * params.ELECTRON_CHARGE * params.ELECTRON_CHARGE));

    % DISPLAY CHANGES
    app.GaugeDebyeElectron.Value = params.ION_DEBYE * 1e6;
    app.GaugeDebyeIon.Value = params.ELECTRON_DEBYE * 1e6;

    % SAVE CHANGES
    app.params = params;
end