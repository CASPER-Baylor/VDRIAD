classdef vdriadLUT2D < vdriadLUT
    %UNTITLED4 Summary of this class goes here
    %   Detailed explanation goes here
    methods
        function result = LookUp(obj,Xq,Yq)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            result = interp2(obj.X,obj.Y,obj.V,Xq,Yq,obj.method);
        end
    end
end