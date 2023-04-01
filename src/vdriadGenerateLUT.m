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
    LUTArray = cell(numel(idx),1);

    for i = 1:length(idx)
        jdx = idx(i);

        % Run Script to generate the corresponding LUT and store
        name = [path '/' listing(jdx).name];
        LUTArray{i} = callFunc(name);

        fprintf('\t%s generated\n',listing(jdx).name);
    end

    fprintf('Saving LUTs Data...\n');
    save('../data/LUTData.mat','LUTArray');
    fprintf('Done.\n')
end

function LUT = callFunc(name)
    % This is just a function that takes care of executing the script. The
    % reason for using a function is that it allows us to keep the
    % workspaces separated so that it doesn't mess up the variables of the
    % current function
    
    run(name);
    close all
end
