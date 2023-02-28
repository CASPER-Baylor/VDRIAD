function vdriadLoadLUT(app)
%LoadLut Loads the lookup table data previously generated to be used as a
%lookup table in the simulation
    load('../data/LUTData.mat','LUTData');

    app.LUTS.LUTDensity = vdriadLUT2D(LUTData);
end