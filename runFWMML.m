%{-1: irrelant label, 1: relevant label, 0: unknown label}
%Based on FWMML, 
%- some difference with paper may be there because of different configurations.
%- Just cleaned up enough so that code is working. Redundant code might
%have been left out.
%Paper: Multi-label learning with missing labels using model coefficients for
%feature relevance, Applied Intelligence, 2023

optmParameter.lambda1   = 10^4; %10^2; %10^1;  % YR missing labels
optmParameter.lambda2   = 10^-7; %10^-7; % W'W embedding
optmParameter.lambda3   = 10^-3; % W Regularization
optmParameter.lambda4   = 10^-4; % R Regularization
optmParameter.lambda5   = 10^-7; % Smoothness in projected input space
optmParameter.lambda6   = 10^-7; % Smoothness in predictions

optmParameter.d1frac    = 0.4; %0.1/0.05 worked great for yeast% fraction of original number of features

optmParameter.etaW      = 10.0; %0.1; %  training rate for W
optmParameter.etaP      = 10.0; %0.1; %  training rate for P
optmParameter.etaR      = 10.0; %0.1; %  training rate for R

optmParameter.maxIter           = 50;   % 50-70
     
%% Model Parametersf
modelparameter.cv_num             = 5;
modelparameter.repetitions        = 1;


addpath(genpath('.'));
clc
model_FWMML.optmParameter  = optmParameter;
model_FWMML.modelparameter = modelparameter;
model_FWMML.tuneThreshold  = 0;% tune the threshold for mlc
fprintf('*** FWMML for multi-label learning with missing labels ***\n');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lambda = {10^-8, 10^-6, 10^-4, 10^-2, 10^0, 10^2, 10^4};
datasets = {'yeast.mat'};
rho = {1, 1, 4, 1, 1, 8, 1};
%missing label rate
misRate = {0.6};

for mr=1:numel(misRate)
    model_FWMML.misRate = misRate{mr}; % missing rate of positive  class labels
%for dc=1:numel(datasets)
for dc=1:numel(lambda)
    clear data target;
    load('yeast.mat');
    optmParameter.rho = rho{dc};
    optmParameter.lambda2 = lambda{dc};
    
    if exist('train_data','var')==1
        data    = [train_data;test_data];
        target  = [train_target,test_target];
    end
    clear train_data test_data train_target test_target

    target(target == 0) = -1;
    %target(target == -1) = 0;

    data      = double (data);
    num_data  = size(data,1);
    temp_data = data + eps;
    %temp_data =
    %temp_data./repmat(sqrt(sum(temp_data.^2,2)),1,size(temp_data,2)); %row
    temp_data = temp_data./repmat(sqrt(sum(temp_data.^2,1)), size(temp_data,1), 1); %column

    if sum(sum(isnan(temp_data)))>0
        temp_data = data+eps;
        %temp_data = temp_data./repmat(sqrt(sum(temp_data.^2,2)),1,size(temp_data,2));
        temp_data = temp_data./repmat(sqrt(sum(temp_data.^2,1)), size(temp_data,1), 1); %column
    end
    %data augmentation with columns of 1 for b's wx+b
    temp_data = [temp_data,ones(num_data,1)];

    rng(0);
    randorder = randperm(num_data);
    cvResult  = zeros(16,modelparameter.cv_num);
%%
    
    %modelparameter.repetitions loop can be used for hyperparameter tuning
    for i = 1:modelparameter.repetitions       
        for j = 1:modelparameter.cv_num
            fprintf('- Repetition - %d/%d,  Cross Validation - %d/%d', i, modelparameter.repetitions, j, modelparameter.cv_num);
            [cv_train_data,cv_train_target,cv_test_data,cv_test_target ] = generateCVSet( temp_data,target',randorder,j,modelparameter.cv_num );

            if model_FWMML.misRate > 0
                 temptarget = cv_train_target;
                 [IncompleteTarget, ~, ~, realpercent]= getIncompleteTarget(cv_train_target, model_FWMML.misRate,1); 
                 fprintf('\n-- Missing rate:%.1f, Real Missing rate %.3f\n',model_FWMML.misRate, realpercent); 
            end
           %% Training
           J = (IncompleteTarget ~= 0);
           optmParameter.J = J;
           modelfwmml  = FWMML( cv_train_data, IncompleteTarget, optmParameter); 
           %% Prediction and evaluation

            Outputs = (cv_test_data * modelfwmml.W)';
            %Outputs = (cv_test_data * modelfwmml.W)';
     
            if model_FWMML.tuneThreshold == 1
                fscore                 = (cv_train_data * modelfwmml.W)';
                %fscore                 = (cv_train_data * modelfwmml.W)';
                [ tau,  currentResult] = TuneThreshold( fscore, cv_train_target', 1, 2);
                %[ tau,  currentResult] = TuneThreshold( fscore, cv_train_target', 0, 2);
                Pre_Labels             = Predict(Outputs,tau);
                %Predict return 0 and 1's not -1 and 1
            else
                Pre_Labels = sign(Outputs);
                %FIX THIS 
                Pre_Labels(Pre_Labels == -1) = 0; %convert -1 to 0, ???
                %Pre_Labels = double(Outputs>=0.5);
            end
            fprintf('-- Evaluation\n');
            tmpResult = EvaluationAll(Pre_Labels,Outputs,cv_test_target');
            cvResult(:,j) = cvResult(:,j) + tmpResult;
        end
    end

cvResult = cvResult./modelparameter.repetitions;
Avg_Result      = zeros(16,2);
Avg_Result(:,1) = mean(cvResult,2);
Avg_Result(:,2) = std(cvResult,1,2);
result =   Avg_Result;
PrintResults(Avg_Result);

end
end