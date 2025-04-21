    projectRoot = getenv('AV_PROJECT_HOME');
nnvRoot = getenv('NNV_ROOT');

nnvRoot = fullfile(nnvRoot);

if ~isfolder(nnvRoot)
    error('NNV folder not found: %s', nnvRoot)
end

% 3) Add NNV (and all subfolders) to your MATLAB path
addpath( genpath(nnvRoot) );

% savepath;

addpath(genpath(fullfile(projectRoot, '/node_verification/functions/')));
addpath(genpath(fullfile(projectRoot, '/node_verification/models/')));

% data = load(fullfile(projectRoot, 'data', 'cora_node.mat'));
data = load(fullfile(projectRoot, 'node_verification', 'CORA_Original', 'reduced_dataset.mat'));
A_full     = data.edge_indices(:,:,1);    
X_full     = data.features(:,:,1);        
y_full     = double(data.labels(:)) + 1;  
numNodes   = size(X_full,1);

rng(2024);
[~, ~, idxTest] = trainingPartitions(numNodes, [0.6 0.3 0.1]);

adjacencyDataTest = A_full(idxTest, idxTest);
featureDataTest   = X_full(idxTest, :);
labelDataTest     = y_full(idxTest);

fprintf('Number of test samples: %d\n', length(idxTest));
fprintf('Feature dimension: %d\n', size(featureDataTest, 2));

%% Verify models

% Study Variables
% seeds = [0,1,2,3,4]; % models
seeds = [0, 1, 2]; % models
num_features = 16;
epsilons = [.00005, .0005, .005]; 

% Verify one model at a time - using regular for loop instead of parfor to avoid file access issues
parfor e = 1:length(epsilons)
    for k = 1:length(seeds)
        % Construct the model path
        modelPath = "cora_node_gcn_" + string(seeds(k) + "_" + string(num_features));
    
        fprintf('Verifying model %s with epsilon %.5f\n', modelPath, epsilons(e));
    
        reach_model_Linf(modelPath, epsilons(e), adjacencyDataTest, featureDataTest, labelDataTest, num_features);

    end
end


% one‑time setup (run in MATLAB interactively)
% mySMTP = 'smtp.gmail.com';            % e.g. Gmail
% myPort = '465';
% myEmail = 'dahle.noah13@gmail.com';
% myPass  = 
% setpref('Internet','SMTP_Server',mySMTP);
% setpref('Internet','E_mail',myEmail);
% setpref('Internet','SMTP_Username',myEmail);
% setpref('Internet','SMTP_Password',myPass);
% props = java.lang.System.getProperties;
% props.setProperty('mail.smtp.auth','true');
% props.setProperty('mail.smtp.socketFactory.class','javax.net.ssl.SSLSocketFactory');
% props.setProperty('mail.smtp.socketFactory.port', myPort);

% sendmail('dahle.noah13@gmail.com', 'Remote MATLAB Done', 'Your script has finished running.');
