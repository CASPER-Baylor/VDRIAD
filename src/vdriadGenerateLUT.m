function vdriadGenerateLUT()
%GENERATELUT is a function that takes care of generating the data necessary
%for the LUTs used in the simulation to work
%   The function loops through all the matlab scripts with the prefix 'LUT'
%   and runs them to generate the data structs necessary to operate the 
%   lookup tables in the simulation
    path = '../scripts';

    % Open the directory and take a look inside
    listing = dir(path);

    % Generate an index to the matlab scripts that have to be run
    idx = false(length(listing),1);

    for i = 1:length(listing)
        name = listing(i).name;
        prefix = split(name,'_');
        extension = split(name,'.');

        if strcmp(prefix{1},'LUT') && strcmp(extension{2},'m')
            idx(i) = true;
        end
    end

    % Run the matlab scripts
    idx = find(idx);
    LUTData = struct('Label',[],'X',[],'Y',[],'V',[]);

    for i = 1:length(idx)
        jdx = idx(i);

        % Run Script to generate the corresponding LUT
        run([path '/' listing(jdx).name])

        % Store the corresponding variables
        LUTData(i).Label = LUTLabel;
        LUTData(i).X = X;
        LUTData(i).Y = Y;
        LUTData(i).V = V;

        fprintf('\t%s generated\n',listing(jdx).name);
    end

    fprintf('Saving LUTs Data...\n');
    save('../data/LUTData.mat','LUTData');
    fprintf('Done.\n')
end