% File Name:        LUT_01.m
% Author:           Jorge A. Martinez-Ortiz
% Due Date:         02.16.2023
% Description:      Lo(ook-up)Ta(ble) ElectronDensity(Power,Pressure)
%                   Matlab script to generate the lookup table of the
%                   electron density as a function of the power and
%                   pressure. The values are taken from nosenkos paper.

T = readtable('../data/Couedel_PhysRevE_105_015210_fig02.xlsx');
T = T(:,["Pressure","Power","Density"]);

% Perform unit conversion
T.Density = T.Density * 1e6;

pressure_vals = unique(T.Pressure);
power_vals = unique(T.Power);

[X,Y] = meshgrid(pressure_vals,power_vals);
V = reshape(T.Density,numel(power_vals),numel(pressure_vals));

LUT.Name = "LUT01";
LUT.Label = ["Pressure";
             "Power";
             "Electron density"];
LUT.Pressure = X;
LUT.Power = Y;
LUT.ElectronDensity = V;

% Run this just to visualize an example of how things would work
% xq = linspace(pressure_vals(1),pressure_vals(end),20);
% yq = linspace(power_vals(1),power_vals(end),20);
% [Xq,Yq] = meshgrid(xq,yq);
% figure
% surf(Xq,Yq,interp2(X,Y,V,Xq,Yq));
% xlabel('Pressure [Pa]');
% ylabel('Power');
% zlabel('$n_e [cm^{-3}$]','Interpreter','latex');










