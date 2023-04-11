% File Name:            LUT_wake.m
% Author:               Jorge A. Martinez-Ortiz
% Date Created:         03.30.2023
% Description:          This is the matlab file that will process the data
%                       of Rahul's paper. This data will be used to obtain
%                       the parameters for the ion wakes as a function of
%                       pressure and power.

% Load and Pre-process data
T = readtable('../data/Banka_10.1088_1361__6587_acbe62_table01.xlsx');
TProc = processData(T);

%% REPLICATE PLOTS

% Fit to the data
x = TProc.Pressure;
xq = linspace(min(x),max(x));

charge = TProc.Qw./TProc.Qd;
length = TProc.L;
debye = TProc.lambdaDe;
chargeDust = TProc.Qd;

% Wake charge
modCharge1 = 'poly1';
[fitCharge1,gofCharge1] = fit(x,charge,modCharge1);

modCharge2 = 'poly2';
[fitCharge2,gofCharge2] = fit(x,charge,modCharge2);

% Length 
modLength = 'poly1';
[fitLength,gofLength] = fit(x,length,modLength);

% Debye
modDebye = fittype('a*x^(-1/2)','independent','x');
[fitDebye,gofDebye] = fit(x,debye,modDebye);

% ChargeDust
modChargeDust = 'poly1';
[fitChargeDust,gofChargeDust] = fit(x,chargeDust,modChargeDust);


% Handle functions for plotting
myData = @(x,y) plot(x,y,'.');
myFit = @(fit) plot(fit,'--');

figure
subplot(4,1,1)
hold on
myData(x,charge)
myFit(fitCharge1);
myFit(fitCharge2);
hold off

ylabel('q_w/Q_d')
strLegend = ['';'R^2='+string([gofCharge1.rsquare;gofCharge2.rsquare])];
legend(strLegend)

subplot(4,1,2)
hold on 
myData(x,length)
myFit(fitLength)
hold off

ylabel('l/\lambda_{De}')
strLegend = ['';'R^2='+string(gofLength.rsquare)];
legend(strLegend)

subplot(4,1,3)
hold on 
myData(x,debye)
myFit(fitDebye)
hold off

ylabel('\lambda_{De}/\mum')
strLegend = ['';'R^2='+string(gofDebye.rsquare)];
legend(strLegend)

subplot(4,1,4)
hold on
myData(x,chargeDust)
myFit(fitChargeDust)
hold off

ylabel('10^4 e^{–}')
xlabel('Pressure [Pa]')
strLegend = ['';'R^2='+string(gofChargeDust.rsquare)];
legend(strLegend)




function TProc = processData(T)
    n = size(T,1);

    % Allocate memory
    Qeff = zeros(n,1);
    dQeff = zeros(size(Qeff));
    
    % Parse through the colums of the table that contain uncertainty 
    % values
    for i = 1:n
        vals = textscan(T.Qeff{i},'%f %*s %f');
        Qeff(i) = vals{1};
        dQeff(i) = vals{2};
    end

    % Convert to metric units
    T.Qeff = Qeff * 1e4;
    T.ni0 = T.ni0 * 1e14;
    T.Qd = T.Qd * 1e4;

    % Save into a new table with an additional column
    dQeff = dQeff * 1e4;
    TProc = [T table(dQeff)];
end