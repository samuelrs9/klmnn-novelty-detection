% Initializations
clc; 
clear; 
close all;
run('../external/lmnn/setpaths3.m');
addpath('..');
addpath('../external/kpca');
addpath('../external/knfst');
addpath('../compared_methods');

% Load the dataset
dt = Datasets('../datasets');
libras = dt.loadLibras();
X = libras.X; 
y = libras.Y;
num_classes = numel(unique(y));

% Experiment configurations
out_dir = 'out_libras';
num_untrained_classes = floor(0.25*num_classes)+1;
training_ratio = 0.8;
random_select_classes = true;
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
klmnn_config.decision_thresholds = linspace(0.5,1.5,klmnn_config.num_decision_thresholds)';
klmnn_config.num_kernels = num_kernels;
klmnn_config.kernel_type = 'gauss';
klmnn_config.reduction_ratio = 1.0; % percent variability explained by principal components
klmnn_config.kernels = linspace(15,20,klmnn_config.num_kernels)';

% KNFST
knfst_config.name = 'knfst';
knfst_config.num_decision_thresholds = num_decision_thresholds;
knfst_config.decision_thresholds = linspace(0.5,0.8,knfst_config.num_decision_thresholds)';
knfst_config.kernel_type = 'gauss';
knfst_config.num_kernels = num_kernels;
knfst_config.kernels = linspace(0.2,0.6,knfst_config.num_kernels)';

% ONE SVM
one_svm_config.name = 'one_svm';
one_svm_config.kernel_type = 'gauss';
one_svm_config.num_kernels = num_kernels;
one_svm_config.kernels = linspace(0.4,1.2,one_svm_config.num_kernels)';

% MULTI SVM
multi_svm_config.name = 'multi_svm';
multi_svm_config.num_decision_thresholds = num_decision_thresholds;
multi_svm_config.decision_thresholds = linspace(0.01,0.8,multi_svm_config.num_decision_thresholds)';
multi_svm_config.kernel_type = 'gauss';
multi_svm_config.num_kernels = num_kernels;
multi_svm_config.kernels = linspace(0.9,1.7,multi_svm_config.num_kernels)';

% KPCA
kpca_config.name = 'kpca';
kpca_config.num_decision_thresholds = num_decision_thresholds;
kpca_config.decision_thresholds = linspace(0.8,1.0,kpca_config.num_decision_thresholds)';
kpca_config.kernel_type = 'gauss';
kpca_config.num_kernels = num_kernels;
kpca_config.kernels = linspace(0.2,1.2,kpca_config.num_kernels)';

% Organize all methods in a cell array
methods  = {knn_config,lmnn_config,klmnn_config,...
  knfst_config,one_svm_config,multi_svm_config,kpca_config};

manager = Manager(X,y,out_dir,num_experiments,...
  num_untrained_classes,training_ratio,random_select_classes,plot_metric);

tutorial = 4;

switch tutorial
  case 2
    % ------------------------------------------------------------------------------------
    % This runs novelty detection experiments for KNN, LMNN and KLMNN based approaches.
    % ------------------------------------------------------------------------------------
    manager.runExperimentsForKnnMethods(methods([1,2,3]),num_knn_args);
  case 2
    % ------------------------------------------------------------------------------------
    % This processes novelty detection results for KNN, LMNN and KLMNN based approaches.
    % ------------------------------------------------------------------------------------
    knn_reports = manager.reportExperimentsForKnnMethods(out_dir,num_knn_args);
  case 3
    % ------------------------------------------------------------------------------------
    % This runs novelty detection experiments for KNFST, ONE SVM, MULTI SVM and KPCA 
    % based approaches.
    % ------------------------------------------------------------------------------------
    manager.runExperiments(methods([4,5,6,7]));    
  case 4
    % ------------------------------------------------------------------------------------
    % This processes novelty detection results for all methods.
    % ------------------------------------------------------------------------------------                
    manager.reportExperiments(out_dir,methods([1,2,3,4,5,6,7]));
end
