% File:             vdriadFormatMain.m
% Author:           Jorge Augusto Martinez-Ortiz
% Date Created:     02.06.2023

function vdriadFormatMain(app)
%vdriadFormatMain Formats the appearance of the main window of the VDRIAD
%application
%   This function will be called at the beginning of the execution of the
%   application (startupFcn) in order to format the title an labels of
%   various elements in the GUI.

    % GRAVITY SPINNER
    app.GravitySpinner.Value = app.params.GRAVITY;
    app.GravitySpinner.Step = 0.1;
    app.GravitySpinner.Limits = [0 100];
    
    % PRESSURE SPINNER
    app.PressureSpinner.Value = app.params.GAS_PRESSURE;
    app.PressureSpinner.Step = 0.1;
    app.PressureSpinner.Limits = [0.66 6];

    % POWER SPINNER
    app.PowerSpinner.Value = app.params.CELL_POWER;
    app.PowerSpinner.Step = 0.1;
    app.PowerSpinner.Limits = [0.1 25];
    
    % DEBYE LENGTH
    app.GaugeDebyeIon.Limits = [0 200];
    app.GaugeDebyeElectron.Limits = [0 200];
    
    % LAMP
    app.PlayLamp.Color = 'red';
    
    % CHECKBOXES
    app.NewseedCheckBox.Value = false;
    
    % TABLE
    strCols = {'N','x','y','z','vx','vy','vz','ax','ay','az'};
    app.UITable.ColumnName = strCols;
end