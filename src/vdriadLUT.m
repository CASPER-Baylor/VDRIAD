classdef (Abstract) vdriadLUT
    %LUT is an abstract class that will allow us to build different
    %lookup tables (1D or 2D) for distinct types of interpolations
    %   Detailed explanation goes here

    properties (Access=protected)
        X
        Y
        V
        label
        method
    end

    methods
        function obj = vdriadLUT(LUTData)
            %LUT Construct an instance of this class
            %   Detailed explanation goes here
            obj.X = LUTData.X;
            obj.Y = LUTData.Y;
            obj.V = LUTData.V;
            obj.label = LUTData.Label;
            obj.method = 'linear';
        end
    end

    methods (Abstract)
        result = LookUp(obj)
    end
end