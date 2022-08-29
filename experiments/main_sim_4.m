% Initializations
clc;
clear;
close all;
run('../external/lmnn/setpaths3.m');
addpath('..');
addpath('../external/kpca');
addpath('../external/knfst');
addpath('../compared_methods');

% Experiment configurations
training_ratio = 0.7;
num_untrained_classes = 1;
random_select_classes = false;
plot_metric = false;

% More experiment configurations
num_experiments = 10;
num_knn_args = 5;
num_decision_thresholds = 40;
num_kernels = 40;

% Hyperparameters
% KNN
knn_config.name = 'knn';
knn_config.num_decision_thresholds = num_decision_thresholds;
knn_config.decision_thresholds = linspace(0.5,1.5,knn_config.num_decision_thresholds)';

% LMNN
lmnn_config.name = 'lmnn';
lmnn_config.num_decision_thresholds = num_decision_thresholds;
lmnn_config.decision_thresholds = linspace(0.5,1.5,lmnn_config.num_decision_thresholds)';

% KLMNN
klmnn_config.name = 'klmnn';
klmnn_config.num_decision_thresholds = num_decision_thresholds;
klmnn_config.decision_thresholds = linspace(0.5,4.5,klmnn_config.num_decision_thresholds)';
klmnn_config.num_kernels = num_kernels;
klmnn_config.kernel_type = 'gauss';
klmnn_config.reduction_ratio = 0.9; % percent variability explained by principal components
klmnn_config.kernels = linspace(0.1,2.0,klmnn_config.num_kernels)';

% KNFST
knfst_config.name = 'knfst';
knfst_config.num_decision_thresholds = num_decision_thresholds;
knfst_config.decision_thresholds = linspace(0.01,1.0,knfst_config.num_decision_thresholds)';
knfst_config.kernel_type = 'gauss';
knfst_config.num_kernels = num_kernels;
knfst_config.kernels = linspace(0.1,1.5,knfst_config.num_kernels)';

% ONE SVM
one_svm_config.name = 'one_svm';
one_svm_config.kernel_type = 'gauss';
one_svm_config.num_kernels = num_kernels;
one_svm_config.kernels = linspace(0.1,1.5,one_svm_config.num_kernels)';

% MULTI SVM
multi_svm_config.name = 'multi_svm';
multi_svm_config.num_decision_thresholds = num_decision_thresholds;
multi_svm_config.decision_thresholds = linspace(0.0,1.5,multi_svm_config.num_decision_thresholds)';
multi_svm_config.kernel_type = 'gauss';
multi_svm_config.num_kernels = num_kernels;
multi_svm_config.kernels = linspace(0.1,1.5,multi_svm_config.num_kernels)';

% KPCA
kpca_config.name = 'kpca';
kpca_config.num_decision_thresholds = num_decision_thresholds;
kpca_config.decision_thresholds = linspace(0.5,0.99,kpca_config.num_decision_thresholds)';
kpca_config.kernel_type = 'poly';
kpca_config.num_kernels = num_kernels;
kpca_config.kernels = linspace(0.1,1.5,kpca_config.num_kernels)';

% Organize all methods in a cell array
methods  = {knn_config,lmnn_config,klmnn_config,...
  knfst_config,one_svm_config,multi_svm_config,kpca_config};

tutorial = 7;

% Variation in the amount of training data
N = [800,100,200,300,400,500,750,1000,1500,2000];

for i=1:numel(N)
  % Variation in the number of spatial dimensions
  if N(i) == 800  
    DIM = 2:20;
  else
    DIM = 10;
  end
  for j=1:numel(DIM)
    % Output diretory
    out_dir = strcat('out_sim_4','/N=',int2str(N(i)),' DIM=',int2str(DIM(j)));        
    
    % Create the synthetic dataset
    num_test_points = 10000;
    data = SyntheticDatasets.uniformDistributions(out_dir,N(i),num_test_points,DIM(j),false);
    xtrain = data.X;
    ytrain = data.y;
    xtest = data.xtest;
    ytest = data.ytest;
  
    % Manager
    manager = Manager(xtrain,ytrain,out_dir,num_experiments,...
      num_untrained_classes,training_ratio,random_select_classes,plot_metric);
    
    switch tutorial
      case 1
        % --------------------------------------------------------------------------------
        % This runs novelty detection experiments for KNN, LMNN and KLMNN based approaches.
        % --------------------------------------------------------------------------------
        manager.runExperimentsForKnnMethods(methods([1,2,3]),num_knn_args);
      case 2
        % --------------------------------------------------------------------------------
        % This processes novelty detection results for KNN, LMNN and KLMNN based approaches.
        % --------------------------------------------------------------------------------
        knn_reports = manager.reportExperimentsForKnnMethods(out_dir,num_knn_args);
      case 3
        % --------------------------------------------------------------------------------
        % This runs novelty detection experiments for KNFST, ONE SVM, MULTI SVM and KPCA
        % based approaches.
        % --------------------------------------------------------------------------------
        manager.runExperiments(methods([4,5,6,7]));
      case 4
        % --------------------------------------------------------------------------------
        % This processes novelty detection results for all methods.
        % --------------------------------------------------------------------------------
        manager.reportExperiments(out_dir,methods([1,2,3,4,5,6,7]));        
      case 5
        % --------------------------------------------------------------------------------
        % This runs evaluations on test sets. [PARCIALMENTE ATUALIZADO]
        % --------------------------------------------------------------------------------
        manager.runEvaluationTests(xtest,ytest,methods([1,2,3,4,5,6,7]),out_dir);
    end
  end
end

out_dir = 'out_sim_4';
switch tutorial
  case 6
    % --------------------------------------------------------------------------------------
    % This loads and processes the data variation experiments 
    % for uniform distribution synthetic dataset.
    % --------------------------------------------------------------------------------------
    N = [100,200,300,400,500,750,1000,1500,2000];
    DIM = 10;  
    manager.reportDataVariationExperiment(methods,out_dir,N,DIM)
  case 7
    % --------------------------------------------------------------------------------------
    % This loads and processes the dimension variation experiments 
    % for uniform distribution synthetic dataset.
    % --------------------------------------------------------------------------------------
    N = 800;
    DIM = 2:20;
    manager.reportDimensionVariationExperiment(methods,out_dir,N,DIM)
end
