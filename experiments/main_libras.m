% Initializations
clc; 
clear; 
close all;
run('../libraries/lmnn/setpaths3.m');
addpath('..');
addpath('../libraries/kpca');
addpath('../libraries/knfst');
addpath('../compared_methods');

% Loads the dataset
dt = Datasets('../datasets');
libras = dt.loadLibras();
X = libras.X; 
y = libras.Y;

out_dir = 'out_libras';
num_classes = numel(unique(y));
max_std = max(std(X));

methods = {'knn','lmnn','klmnn','knfst','one_svm','multi_svm','kpca'};

num_experiments = 5;
num_untrained_classes = floor(0.25*num_classes);
training_ratio = 0.8;
random_select_classes = true;
plot_metric = false;

% Hyperparameters
%K = 2;
%kappa = 1;

% This sets hyperparameter search ranges for the kernel parameter and decision threshold
% KNN
knn_par.num_decision_thresholds = 10;
knn_par.decision_thresholds = linspace(0.5,1.5,knn_par.num_decision_thresholds)';

% LMNN
lmnn_par.num_decision_thresholds = 50;
lmnn_par.decision_thresholds = linspace(0.5,1.5,lmnn_par.num_decision_thresholds)';

% KLMNN
klmnn_par.num_decision_thresholds = 50;
klmnn_par.decision_thresholds = linspace(0.5,1.5,klmnn_par.num_decision_thresholds)';
klmnn_par.num_kernels = 20;
klmnn_par.kernel_type = 'gauss';
klmnn_par.kernels = linspace(15,20,klmnn_par.num_kernels)';

% KNFST
knfst_par.num_decision_thresholds = 50;
knfst_par.decision_thresholds = linspace(0.5,0.8,knfst_par.num_decision_thresholds)';
knfst_par.kernel_type = 'gauss';
knfst_par.num_kernels = 20;
knfst_par.kernels = linspace(0.2,0.6,knfst_par.num_kernels)';

% ONE SVM
one_svm_par.kernel_type = 'gauss';
one_svm_par.num_kernels = 50;
one_svm_par.kernels = linspace(0.4,1.2,one_svm_par.num_kernels)';

% MULTI SVM
multi_svm_par.num_decision_thresholds = 50;
multi_svm_par.decision_thresholds = linspace(0.01,0.8,multi_svm_par.num_decision_thresholds)';
multi_svm_par.kernel_type = 'gauss';
multi_svm_par.num_kernels = 20;
multi_svm_par.kernels = linspace(0.9,1.7,multi_svm_par.num_kernels)';

% KPCA
kpca_par.num_decision_thresholds = 50;
kpca_par.decision_thresholds = linspace(0.8,1.0,kpca_par.num_decision_thresholds)';
kpca_par.kernel_type = 'gauss';
kpca_par.num_kernels = 20;
kpca_par.kernels = linspace(0.2,1.2,kpca_par.num_kernels)';

%hyperparameters = struct('knn',knn_par,'lmnn',lmnn_par,'klmnn',klmnn_par,...
%  'knfst',knfst_par,'one_svm',one_svm_par,'multi_svm',multi_svm_par,'kpca',kpca_par);

hyperparameters = {knn_par,lmnn_par,klmnn_par,knfst_par,one_svm_par,multi_svm_par,kpca_par};

manager = Manager(X,y,out_dir,num_experiments,...
  num_untrained_classes,training_ratio,random_select_classes,plot_metric);

tutorial = 2;

switch tutorial
  case 1
    % ------------------------------------------------------------------------------------
    % It runs novelty detection experiments for KNN, LMNN and KLMNN based approaches.
    % ------------------------------------------------------------------------------------
    manager.runExperimentsForKnnMethods(methods([1]),hyperparameters([1,2,3]));  
  case 2
    % ------------------------------------------------------------------------------------
    % It processes novelty detection results for KNN, LMNN and KLMNN based approaches.
    % ------------------------------------------------------------------------------------
    manager.reportExperimentsForKnnMethods(out_dir,hyperparameters);
  case 3
    % ------------------------------------------------------------------------------------
    % Runs novelty detection experiments for KNFST, ONE SVM, MULTI SVM and KPCA 
    % based approaches.
    % ------------------------------------------------------------------------------------
    manager.runExperiments(methods([4,5,6,7]));    
  case 4
    % ------------------------------------------------------------------------------------
    % Process novelty detection results for KNFST, ONE SVM, MULTI SVM and KPCA 
    % based approaches.
    % ------------------------------------------------------------------------------------    
    manager.reportExperimentResults(out_dir);
end
