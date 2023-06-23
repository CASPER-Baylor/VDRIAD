function vdriadDrawDust(app)
%UNTITLED5 Summary of this function goes here
%   The function that will draw the particles to the screen
    % Ultimately we want to combine the two functions into one and
    % have them update whenever the first frame is not set

    dustSize = 50;
    x = app.Dust.Position.Host.x;
    y = app.Dust.Position.Host.y;
    z = app.Dust.Position.Host.z;

    if app.firstFrame
        hold(app.UIAxes,'on')
        % Draw dust particles
        app.dustAxes = scatter3(app.UIAxes,...
                               x,...
                               y,...
                               z, ...
                               dustSize,'.k');

        % Draw wake particles
        app.wakeAxes = scatter3(app.UIAxes,...
                               x,...
                               y,...
                               z-(app.Parameters.WAKE_LENGTH*10),...
                               5,'.r');
        hold(app.UIAxes,'off')

        % Flag first frame as false
        app.firstFrame = false;
    else
        % Update dust particles
        app.dustAxes.XData = x;
        app.dustAxes.YData = y;
        app.dustAxes.ZData = z;

        % Update wake particles
        app.wakeAxes.XData = x;
        app.wakeAxes.YData = y;
        app.wakeAxes.ZData = z-(app.Parameters.WAKE_LENGTH*10);
    end  

    label_time = sprintf("t = %.2fs",app.TOTAL_TIME);
    title(app.UIAxes,['GEC RF Reference Cell 3D View ' label_time])
end