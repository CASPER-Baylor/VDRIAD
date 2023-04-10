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

% Wake charge
p = polyfit(x,charge,1);
chargeQ1 = polyval(p,xq);

p = polyfit(x,charge,2);
chargeQ2 = polyval(p,xq);

% Length 
p = polyfit(x,length,1);
lengthQ1 = polyval(p,xq);

% Debye
f = fittype('a*x^(-1/2)','independent','x');
[debyeQ1,gofDebye] = fit(x,debye,f);

% Plot as a function of pressure
myData = @(x,y) plot(x,y,'.');
myFit = @(x,y) plot(x,y,'--');

figure
subplot(3,1,1)
hold on
myData(x,charge)
myFit(xq,chargeQ1)
myFit(xq,chargeQ2)
hold off
ylabel('q_w/Q_d')

subplot(3,1,2)
hold on 
myData(x,length)
myFit(xq,lengthQ1)
hold off
ylabel('l/\lambda_{De}')

subplot(3,1,3)
hold on 
myData(x,debye)
plot(debyeQ1)
hold off
ylabel('\lambda_{De}/\mum')
xlabel('Pressure [Pa]')

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