%% Create visualizations for computed L_inf results

% We are interested in:
% 1) How many complete molecules are completely robustly verified (all atoms in a moolecule)?
% 2) How many atoms are robustly verified?

%% Process results for each model independently

% seeds = [0,1,2,3,4]; % models
seeds = [1]; % models
epsilon = [0.0005] %; 0.01; 0.02; 0.05];
eN = length(epsilon);

projectRoot = getenv('AV_PROJECT_HOME');
baseDir = fullfile(projectRoot,'node_verification','CORA_Original','verification_results');
matDir = fullfile(baseDir, 'mat_files');

% make sure the dirs exist
if ~exist(matDir,'dir'),    mkdir(matDir);    end
if ~exist(baseDir,'dir'),   mkdir(baseDir);   end

% Verify one model at a time
for m=1:length(seeds)

    % get model
    modelPath = "cora_node_gcn_"+string(seeds(m));
    
    % initialize vars
    samples = zeros(eN,4);

    for k = 1:eN
        % Load data one at a time
        epsStr = sprintf('%.4f', epsilon(k));   % -> '0.0005'

        fname = fullfile(matDir, ...
        sprintf("verification_results_%s_eps_%s.mat", modelPath, epsStr));
        if ~isfile(fname)
            error("Cannot find %s", fname);
        end
        S = load(fname, 'results','targets');

        if ~isfile(fname)
            error("Cannot find %s", fname);
        end
        S = load(fname, 'results','targets');

        % 3) tally your results
        N = numel(S.targets);
        for i = 1:N
            r = S.results(i);
            samples(k,1) = samples(k,1) + sum(r==1);  % robust
            samples(k,2) = samples(k,2) + sum(r==2);  % unknown
            samples(k,3) = samples(k,3) + sum(r==0);  % not robust
            samples(k,4) = samples(k,4) + numel(r);   % total atoms
        end
    end

    % Save summary
    save("verification_results/mat_files/summary_results_"+modelPath+".mat", "samples");

    model = load("models/"+modelPath+".mat");
    
    % Create table with these values
    fileID = fopen("verification_results/summary_results_"+modelPath+".txt",'w');
    fprintf(fileID, 'Summary of robustness results of CORA gnn model with accuracy = %.4f \n\n', model.testAcc);
    fprintf(fileID,'                 Compromised \n');
    fprintf(fileID, 'Epsilon | Robust  Unknown  Not Rob.  N \n');
    fprintf(fileID, '  0.0005 | %.3f    %.3f   %.3f   %d \n', samples(1,1)/samples(1,4), samples(1,2)/samples(1,4), samples(1,3)/samples(1,4), samples(1,4));
    % fprintf(fileID, '   0.01 | %.3f    %.3f   %.3f   %d \n', samples(2,1)/samples(2,4), samples(2,2)/samples(2,4), samples(2,3)/samples(2,4), samples(2,4));
    % fprintf(fileID, '   0.02 | %.3f    %.3f   %.3f   %d \n', samples(3,1)/samples(3,4), samples(3,2)/samples(3,4), samples(3,3)/samples(3,4), samples(3,4));
    % fprintf(fileID, '   0.05 | %.3f    %.3f   %.3f   %d \n', samples(4,1)/samples(4,4), samples(4,2)/samples(4,4), samples(4,3)/samples(4,4), samples(4,4));
    fclose(fileID);

end
