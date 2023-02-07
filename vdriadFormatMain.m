% File:             vdriadFormatMain
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
    app.IonDebyeGauge.Limits = [0 200];
    app.ElectronDebyeGauge.Limits = [0 200];
    
    % LAMP
    app.PlayLamp.Color = 'red';
    
    % CHECKBOXES
    app.MovieCheckBox.Value = true;
    app.NewseedCheckBox.Value = false;
    
    % TABLE
    %app.UITable.ColumnName = {'ax','ay','az'};
end