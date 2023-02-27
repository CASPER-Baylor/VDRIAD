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
    app.GravityNkgSpinner.Value = app.GRAVITY;
    app.GravityNkgSpinner.Step = 0.1;
    app.GravityNkgSpinner.Limits = [0 100];
    
    % PRESSURE SPINNER
    app.PressurePaSpinner.Value = app.GAS_PRESSURE;
    app.PressurePaSpinner.Step = 0.1;
    app.PressurePaSpinner.Limits = [0.1 200];
    
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