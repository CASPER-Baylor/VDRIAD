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
    app.GravitySpinner.Value = app.Parameters.GRAVITY;
    app.GravitySpinner.Step = 0.1;
    app.GravitySpinner.Limits = [0 100];
    
    % PRESSURE SPINNER
    app.PressureSpinner.Value = app.Parameters.GAS_PRESSURE;
    app.PressureSpinner.Step = 0.1;
    app.PressureSpinner.Limits = [0.66 6];

    % POWER SPINNER
    app.PowerSpinner.Value = app.Parameters.CELL_POWER;
    app.PowerSpinner.Step = 0.1;
    app.PowerSpinner.Limits = [1 20];
    
    % LAMP
    app.PlayLamp.Color = 'red';
    
    % CHECKBOXES
    app.NewseedCheckBox.Value = false;
    
    % TABLES
    % Dynamics table
    strCols = {'N','x','y','z','vx','vy','vz','ax','ay','az'};
    app.UITable.ColumnName = strCols;

    % Parameters table
    strCols = {'Parameters','Value'};
    strParams = {'$\lambda_i$',...
                 '$\lambda_e$',...
                 '$n_e,n_i$'};
    valParams = zeros(3,1);

    T = table(strParams',valParams,'VariableNames',strCols);
    app.UITable2.Data = T;
    s = uistyle('Interpreter','latex','HorizontalAlignment','center');
    addStyle(app.UITable2,s,'column','Parameters');

    %s = uistyle('HorizontalAlignment','center');
    %addStyle(app.UITable2,s,'column','Value');
    
end