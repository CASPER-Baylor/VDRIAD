% File Name:        LUT_01.m
% Author:           Jorge A. Martinez-Ortiz
% Due Date:         02.16.2023
% Description:      Table Lookup to find the electric potential as a
%                   function of the height above the lower electrode

%% READ DATA
clear

% Read the tables containing the data
TPotential = readtable('../data/Couedel_PhysRevE_105_015210_fig08.xlsx');
TPotential = TPotential(:,["Pressure","Power","z","V"]);

TSheath = readtable('../data/Couedel_PhysRevE_105_015210_fig08_sheathHeight.csv');
TSheath = TSheath(:,["Pressure","Power","z0"]);


%% PROCESS DATA TO OBTAIN QUADRATIC FITS
% Array to store the coefficient of the parabola
beta = zeros(numel(TSheath.z0),1);

% Identify the values at which powers and pressures at which measurements
% were taken
pressureVals = unique(TPotential.Pressure);
powerVals = unique(TPotential.Power);

color = 'bm';
ylimPotential = [-150 0];

figure
for i = 1:numel(pressureVals)
    % Current pressure
    pressure = pressureVals(i);

    for j = 1:numel(powerVals)
        % Current power
        power = powerVals(j);
        
        % Index to Data
        idx = (TPotential.Pressure == pressure) & (TPotential.Power == power);
        z = TPotential.z(idx);
        V = TPotential.V(idx);

        idy = (TSheath.Pressure == pressure) & (TSheath.Power == power);
        z0 = TSheath.z0(idy);

        % Fit parabola and determine query points
        xq = linspace(0,z0,100);
        par.b0 = fitParabola(z,V,0);
        beta(idy) = par.b0;
        par.x0 = 0;
        
        % ===== RECREATE ORIGINAL PLOTS
        subplot(3,3,i)
        hold on
        plotFit(z,V,xq,color(j),par);
        hold off
        xline(z0,':r')

        title(sprintf("$p_{Ar}=%0.1f$ Pa",pressure),'Interpreter','latex')
        if i == 1; ylabel('$V_p(V)$','Interpreter','latex'); end
        ylim(ylimPotential);

        % Shift the orgin to be located at the bottom of the cell
        z = z - z0;
        z = -z;

        % Fit parabola
        xq = linspace(0,z0,100);
        par.b0 = fitParabola(z,V,z0);
        par.x0 = z0;
        
        % ===== CREATE SHIFTED PLOTS AND NEW PARABOLAS
        subplot(3,3,i+3)
        hold on
        plotFit(z,V,xq,color(j),par);
        hold off
        xline(z0,':r')

        if i == 1; ylabel('$V_p(V)$ (shifted)','Interpreter','latex'); end
        ylim(ylimPotential);

        % ===== CREATE PLOTS OF THE ELECTRIC FIELDS
        subplot(3,3,i+6)
        hold on
        plotElectricField(xq,color(j),par);
        hold off

        if i == 1; ylabel('$E(V/cm)$','Interpreter','latex'); end
        xlabel('$z(cm)$','Interpreter','latex')
        ylim([-50 0])
    end
end

%% ORGANIZE DATA TO OBTAIN 2D INTERPOLATION
%[X,Y] = meshgrid(pressure_vals,power_vals);
%V = reshape(T.Density,numel(power_vals),numel(pressure_vals));

%LUTLabel = 'LUT01';

% Run this just to visualize an example of how things would work
% xq = linspace(pressure_vals(1),pressure_vals(end),20);
% yq = linspace(power_vals(1),power_vals(end),20);
% [Xq,Yq] = meshgrid(xq,yq);
% figure
% surf(Xq,Yq,interp2(X,Y,V,Xq,Yq));
% xlabel('Pressure [Pa]');
% ylabel('Power');
% zlabel('$n_e [cm^{-3}$]','Interpreter','latex');

function plotFit(x,y,xq,color,par)
    % Evaluate parabola at query points
    yq = parabola(xq,par);

    hold on
    scatter(x,y,100,['.' color]);
    scatter(xq(end),yq(end),'xr');
    plot(xq,yq,['-' color]);
    hold off
end

function plotElectricField(xq,color,par)
    % Evaluate parabola at query points
    yq = parabola(xq,par);

    dx = xq(2)-xq(1);
    E = -diff(yq,1) / dx;

    plot(xq(2:end),E,['-' color]);
end

function yq = parabola(x,par)
    yq = par.b0 * (x-par.x0).^2;
end

% Fits a parabola centered at x0
function b0 = fitParabola(x,y,x0)
    xEff = x-x0;
    b0 = sum(y .* xEff.^2)/sum(xEff.^4);
end










