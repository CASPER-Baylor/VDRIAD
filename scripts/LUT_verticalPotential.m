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

% X: Pressure
% Y: Power
[X,Y] = meshgrid(pressureVals,powerVals);

% V: This is the height of the sheath, but also the point x0 around which
%    the parabola is centerd
V = reshape(TSheath.z0,numel(powerVals),numel(pressureVals));

% W: This is the parameter that defines the parabola equivalent to 
%    b0 in the expression b0(x-x0)^2
W = reshape(beta,numel(powerVals),numel(pressureVals));

% Define query points for interpolation
xq = linspace(pressureVals(1),pressureVals(end),20);
yq = linspace(powerVals(1),powerVals(end),20);
[Xq,Yq] = meshgrid(xq,yq);

figure
% Plot sheath height
subplot(1,2,1)
surf(Xq,Yq,interp2(X,Y,V,Xq,Yq));

title('Sheath height as function of Pressure and Power')
xlabel('Pressure [Pa]')
ylabel('Power [W]')
zlabel("$h_s[cm]$",'Interpreter','latex','FontSize',20)

% Plot the slope of the electric field
subplot(1,2,2)
surf(Xq,Yq,interp2(X,Y,(-2 *W),Xq,Yq));

title('Electric field steepness as a function of Pressure and Power')
xlabel('Pressure [Pa]')
ylabel('Power [W]')
zlabel("$\frac{dE}{dz}$ [$\frac{V}{cm^2}$]",'Interpreter','latex','FontSize',20)



%LUTLabel = 'LUT01';

% Run this just to visualize an example of how things would work

% figure

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










