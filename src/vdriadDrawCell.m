% File Name:            vdriadDrawCell.m
% Author:               Jorge A Martinez-Ortiz
% Date Created:         02.26.2023

function vdriadDrawCell(axes,params)
%DRAWCELL takes care of drawing the GEC RF cell to the screen 
    % Set axis limits
    axes.XLim = [-1 1] * (1.2) * params.CELL_RADIUS;
    axes.YLim = [-1 1] * (1.2) * params.CELL_RADIUS;
    axes.ZLim = [0 1] * (1.2) * params.CELL_HEIGHT;
    
    % Draw cell electrodes and sheath
    theta = linspace(0,2*pi,30);
    x = params.CELL_RADIUS * cos(theta);
    y = params.CELL_RADIUS * sin(theta);

    hold(axes,'on')
        % Draw lower electrode 
        z = zeros(numel(theta),1);
        plot3(axes,x,y,z,'k');
    
        % Draw upper electrode
        z = repmat(params.CELL_HEIGHT,[numel(theta) 1]);
        plot3(axes,x,y,z,'k');
    
        % Draw sheath
        z = repmat(params.SHEATH_HEIGHT,[numel(theta) 1]);
        plot3(axes,x,y,z,'m');
    hold(axes,'off')

    % Label the axes
    xlabel(axes,'x [m]')
    ylabel(axes,'y [m]')
    zlabel(axes,'z [m]')
    
    view(axes,3)
end