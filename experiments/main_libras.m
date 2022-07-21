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

% Manager
Manager = {'knn','lmnn','klmnn','knfst','one_svm','multi_svm','kpca'};

num_experiments = 10;
untrained_classes = floor(0.25*num_classes);
training_ratio = 0.8;
plot_metric = false;

% Hyperparameters
% Para KNN, LMNN e KLMNN
K = 2;
kappa = 1;

% Define intervalos de busca de parāmetros de kernel e thresholds
% KNN
knn_par.num_thresholds = 50;
knn_par.threshold = linspace(0.5,1.5,knn_par.num_thresholds)';

% LMNN
lmnn_par.num_thresholds = 50;
lmnn_par.threshold = linspace(0.5,1.5,lmnn_par.num_thresholds)';

% KLMNN
klmnn_par.num_thresholds = 50;
klmnn_par.threshold = linspace(0.5,1.5,klmnn_par.num_thresholds)';
klmnn_par.num_kernels = 20;
klmnn_par.kernel_type = 'gauss';
klmnn_par.kernel = linspace(15,20,klmnn_par.num_kernels)';

% KNFST
knfst_par.num_thresholds = 50;
knfst_par.threshold = linspace(0.5,0.8,knfst_par.num_thresholds)';
knfst_par.kernel_type = 'gauss';
knfst_par.num_kernels = 20;
knfst_par.kernel = linspace(0.2,0.6,knfst_par.num_kernels)';

% ONE SVM
one_svm_par.kernel_type = 'gauss';
one_svm_par.num_kernels = 50;
one_svm_par.kernel = linspace(0.4,1.2,one_svm_par.num_kernels)';

% MULTI SVM
multi_svm_par.num_thresholds = 50;
multi_svm_par.threshold = linspace(0.01,0.8,multi_svm_par.num_thresholds)';
multi_svm_par.kernel_type = 'gauss';
multi_svm_par.num_kernels = 20;
multi_svm_par.kernel = linspace(0.9,1.7,multi_svm_par.num_kernels)';

% KPCA
kpca_par.num_thresholds = 50;
kpca_par.threshold = linspace(0.8,1.0,kpca_par.num_thresholds)';
kpca_par.kernel_type = 'gauss';
kpca_par.num_kernels = 20;
kpca_par.kernel = linspace(0.2,1.2,kpca_par.num_kernels)';

parameters = {knn_par,lmnn_par,klmnn_par,knfst_par,one_svm_par,multi_svm_par,kpca_par};

tutorial = 1;

switch tutorial
  case 1
    % ------------------------------------------------------------------------------------
    % Runs novelty detection experiments for KNN, LMNN and KLMNN based methods.
    % ------------------------------------------------------------------------------------
    Manager.runExperimentsForKnnMethods(X,y,Manager([1,2,3]),parameters,out_dir,...
                       num_experiments,num_classes,untrained_classes,training_ratio,plot_metric);  
  case 2
    % ------------------------------------------------------------------------------------
    % Process novelty detection results for KNN, LMNN, and KLMNN based methods.
    % ------------------------------------------------------------------------------------
    Manager.reportExperimentsForKnnMethods(out_dir);
  case 3
    % ------------------------------------------------------------------------------------
    % Runs novelty detection experiments for KNFST, ONE SVM, MULTI SVM and KPCA 
    % based methods.
    % ------------------------------------------------------------------------------------
    Manager.runExperiments(X,y,Manager([4,5,6,7]),parameters,out_dir,num_experiments,...
                            num_classes,untrained_classes,training_ratio,K,kappa,plot_metric);    
  case 4
    % ------------------------------------------------------------------------------------
    % Process novelty detection results for KNFST, ONE SVM, MULTI SVM and KPCA 
    % based methods.
    % ------------------------------------------------------------------------------------    
    Manager.reportExperimentResults(Manager,out_dir,K,kappa);
end