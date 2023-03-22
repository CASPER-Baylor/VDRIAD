% File Name:        LUT_01.m
% Author:           Jorge A. Martinez-Ortiz
% Due Date:         02.16.2023
% Description:      Table Lookup to find the electric potential as a
%                   function of the height above the lower electrode

T = readtable('../data/Couedel_PhysRevE_105_015210_fig08.csv');
zShifted = T.z;
TOffset = readtable('../data/Couedel_PhysRevE_105_015210_fig08_sheathHeight.csv');


pressureVals = unique(T.Pressure);
powerVals    = unique(T.Power);
zVals        = unique(T.z);

figure
for i = 1:numel(pressureVals)
    pressure = pressureVals(i);

    % Get indices to the corresponding values
    idx = T.Pressure == pressure;
    jdx = TOffset.Pressure == pressure;

    % Get the data
    V = reshape(T(idx,:).V,3,2);
    sheathHeight = TOffset.z0(jdx);

    % Query points
    n = 100;
    Vq = zeros(100,2);
    zq = linspace(0,10,100);

    % Interpolate for each series
    p = polyfit(zVals,V(:,1),2);
    Vq(:,1) = polyval(p,zq);
    p = polyfit(zVals,V(:,2),2);
    Vq(:,2) = polyval(p,zq);

    % Plot sample vals
    subplot(3,3,i)
    ax_data = scatter(zVals,V,150,'.');
    title(sprintf("$p_{Ar}=%.1f$ Pa ",pressure),'Interpreter','latex',...
                                                'FontSize',15)
    axis square
    xlim([0 10])
    ylim([-150 30])

    if i == 1
            ylabel('$V_p(V)$','Interpreter','latex')
    end

    % Plot quadratic fit
    hold on
    ax_fit = plot(zq,Vq,'--');
    ax_fit(1).Color = ax_data(1).CData;
    ax_fit(2).Color = ax_data(2).CData;
    hold off

    % Plot sheath heights
    xline(sheathHeight(1),'-','Color',ax_data(1).CData);
    xline(sheathHeight(2),'-','Color',ax_data(2).CData);

    % Legend
    legend('5W','25W','5W fit','25W fit')

    % Fix the data by shifting coordinate system to be centered 
    % At the bottom of the plate and not the sheath
    for k = 1:numel(powerVals)
        power = powerVals(k);
        kdx = T.Power == power;

        % Index pointing to the corresponding pressure and the
        % corresponding power
        index = idx & kdx;

        % Shift z coordinate axis
        zShifted(index) = zShifted(index) - sheathHeight(k);
        zShifted(index) = -zShifted(index);

        % Perform quadratic fit
        zq = linspace(0,sheathHeight(k),n);
        p = polyfit(zShifted(index),V(:,k),2);
        Vq(:,k) = polyval(p,zq);

        % Plot Shifted measurements
        subplot(3,3,i+3)
        hold on
        ax_data = scatter(zShifted(index),V(:,k),150,'.');
        ax_fit = plot(zq,Vq(:,k),'--');

        ax_fit.Color = ax_data.CData;
        hold off
    end
    if i == 1
        ylabel('$V_p(V)$','Interpreter','latex')
    end
    yline(0,'-r')

    for k = 1:numel(powerVals)
        % Plot Electric field
        subplot(3,3,i+6)
        dV = Vq(2,k)-Vq(1,k);
        E = -diff(Vq(:,k),1,1) / dV;

        hold on
        plot(zq(2:end),E)
        hold off

        xlabel('z(mm)','Interpreter','latex')
    end
    if i == 1
        ylabel('$E(V/mm)$','Interpreter','latex')
    end
end



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









