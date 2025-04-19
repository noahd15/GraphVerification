% verify_AllReachSets.m

projectRoot = getenv('AV_PROJECT_HOME');
matDir = fullfile(projectRoot, 'node_verification','CORA', verification_results','mat_files');

epsilon = [0.0005];
seeds   = [1];

for m = 1:numel(seeds)
  modelPath = "cora_node_gcn_" + string(seeds(m));

  for e = 1:numel(epsilon)
    epsStr = sprintf('%.4f', epsilon(e));
    inFile = fullfile(matDir, sprintf( ...
             "verified_nodes_%s_eps_%s.mat", modelPath, epsStr));

    S = load(inFile, "outputSets", "targets", "rT");
    N = numel(S.outputSets);

    % Preallocate
    results = zeros(N,1);

    parfor i = 1:N
      Ynode     = S.outputSets{i};    % Star in R^C
      trueLabel = S.targets{i};       % scalar in 1…C
      results(i)= verifySample(Ynode, trueLabel);
    end

    % --- save results (outside parfor) ---
    outFile = fullfile(matDir, sprintf( ...
             "verification_results_%s_eps_%s.mat", modelPath, epsStr));
    rT      = S.rT;
    targets = S.targets;
    save(outFile, "results", "rT", "targets", "-v7");
    fprintf("Wrote → %s\n", outFile);
  end
end

% ----------------------------------------------------------------
function res = verifySample(Ystar, trueLabel)
    % pull bounds
    [lb,~] = Ystar.getRanges();   % 1×C
    C      = numel(lb);

    % build half-spaces y_true ≥ y_j for all j≠trueLabel
    Hs = HalfSpace.empty;
    for j = 1:C
      if j==trueLabel, continue; end
      a = zeros(1,C);
      a(j)         =  1;   % + y_j
      a(trueLabel) = -1;   % – y_true
      Hs(end+1)    = HalfSpace(a, 0);
    end

    % single NNV verification call
    res = verify_specification(Ystar, Hs);
end




% projectRoot = '/home/kendra/Code/other/Verification/AV_Project';
% matDir = fullfile(projectRoot, 'node_verification', 'CORA', ...
%                   'verification_results', 'mat_files');

% %% Verify the robustness reach sets of all models
% % Adjust epsilon and seeds as needed
% epsilon = [0.0005] %; 0.01; 0.02; 0.05];
% seeds = [1];

% parfor m = 1:length(seeds)

%     modelPath = "cora_node_gcn_" + string(seeds(m));
    
%     for k = 1:length(epsilon)
%         % Load outputs (must match how you saved them)
%         epsStr = sprintf('%.4f',epsilon(k));

%         fname  = fullfile(matDir, sprintf("verified_nodes_%s_eps_%s.mat", modelPath, epsStr));
%         fprintf("Loading → %s (exists=%d bytes=%d)\n", fname, isfile(fname), dir(fname).bytes);
%         rdata = load(fname);
%         % Check verification result
%         results = cell(size(rdata.outputSets));
%         for i = 1:length(rdata.outputSets)
%             Y = rdata.outputSets{i};
%             lbl = rdata.targets{i};
%             results{i} = verifySample(Y, lbl);
%         end
        
%         % Save results
%        parsave(modelPath, epsStr, results, rdata.outputSets, rdata.rT, rdata.targets, matDir);
%     end

% end

% function results = verifySample(X, target)
%     % Generic sample-level verification
%     Nsample = size(target,1);
%     results = 3 * ones(Nsample,1);
%     for i = 1:Nsample
%         matIdx = zeros(1,Nsample);
%         matIdx(i) = 1;
%         if iscell(X)
%             X = X{1};
%         end
%         Y = X.affineMap(matIdx, []);
%         Y = Y.toStar; 
%         sampleLabel = target(i,:);
%         sampleHs = label2Hs(sampleLabel);
%         res = verify_specification(Y, sampleHs);
%         if res == 2
%             res = checkViolated(Y, sampleLabel);
%         end
%         results(i) = res;
%     end
% end

% function res = checkViolated(Set, label)
%     target = getLabelIndex(label);
%     [lb, ub] = Set.getRanges;
%     maxTarget = ub(target);
%     if any(lb > maxTarget)
%         res = 0;  % falsified
%     else
%         res = 2;  % unknown
%     end
% end

% function Hs = label2Hs(label)
%     % First determine the actual dimension of your model output
%     % For debugging, add this at the beginning of verifySample:
%     % fprintf('Star dimension: %d\n', Y.dim);
    
%     % Then set outSize to match your actual model output dimension
%     outSize = 272*7; 
%     target = getLabelIndex(label);

%     % Create verification constraints matching your model's output dimension
%     G = -eye(outSize);
%     G(:, target) = 1;
%     g = zeros(outSize,1);

%     Hs = [];
%     for i = 1:length(g)
%         Hs = [Hs; HalfSpace(G(i,:), g(i))];
%     end
% end

% function index = getLabelIndex(label)
%     % CORA dataset labels (7 classes)
%     % 1: 'Case_Based'
%     % 2: 'Genetic_Algorithms'
%     % 3: 'Neural_Networks'
%     % 4: 'Probabilistic_Methods'
%     % 5: 'Reinforcement_Learning'
%     % 6: 'Rule_Learning'
%     % 7: 'Theory'
%     if isnumeric(label)
%         index = label;
%     else
%         switch string(label)
%             case {'Case_Based', 'case_based'}
%                 index = 1;
%             case {'Genetic_Algorithms', 'genetic_algorithms'}
%                 index = 2;
%             case {'Neural_Networks', 'neural_networks'}
%                 index = 3;
%             case {'Probabilistic_Methods', 'probabilistic_methods'}
%                 index = 4;
%             case {'Reinforcement_Learning', 'reinforcement_learning'}
%                 index = 5;
%             case {'Rule_Learning', 'rule_learning'}
%                 index = 6;
%             case {'Theory', 'theory'}
%                 index = 7;
%             otherwise
%                 index = 1; % fallback
%         end
%     end
% end

% function parsave(modelPath, epsStr, results, outputSets, rT, targets, matDir)
%     fname = fullfile(matDir, sprintf("verified_nodes_%s_eps_%s.mat", modelPath, epsStr));
%     save(fname, "results","outputSets","rT","targets");
% end

