%% Inicializa��es
clc; clear; close all;
run('libraries/lmnn/setpaths3.m');
addpath('libraries/kpca');
addpath('libraries/knfst');
addpath(pwd);

%% CARREGA A BASE
libras = Bases.loadBodyPoses;
X = libras.X; 
y = libras.Y;

out_dir = 'body';
n_classes = numel(unique(y));
max_std = max(std(X));

%% CONFIGURA��ES
methods = {'knn','lmnn','klmnn','knfst','one_svm','multi_svm','kpca'};

n_experiments = 10;
untrained_classes = floor(0.5*n_classes);
% n_experiments = 3*n_classes;
% untrained_classes = 1;
training_ratio = 0.8;
view_plot_metric = false;

%% HIPERPAR�METROS
% Para KNN, LMNN e KLMNN
K = 1;
kappa = 1;

% Define intervalos de busca de par�metros de kernel e thresholds
% KNN
knn_par.n_thresholds = 50;
knn_par.threshold = linspace(0.1,1.5,knn_par.n_thresholds)';

% LMNN
lmnn_par.n_thresholds = 50;
lmnn_par.threshold = linspace(0.1,1.5,lmnn_par.n_thresholds)';

% KLMNN
klmnn_par.n_thresholds = 50;
klmnn_par.threshold = linspace(0.8,3.7,klmnn_par.n_thresholds)';
klmnn_par.n_kernels = 20;
klmnn_par.kernel_type = 'gauss';
klmnn_par.kernel = linspace(0.6,2,klmnn_par.n_kernels)';

% KNFST
knfst_par.n_thresholds = 50;
knfst_par.threshold = linspace(0.1,0.7,knfst_par.n_thresholds)';
knfst_par.kernel_type = 'gauss';
knfst_par.n_kernels = 20;
knfst_par.kernel = linspace(0.2,3.5,knfst_par.n_kernels)';

% ONE SVM
one_svm_par.kernel_type = 'gauss';
one_svm_par.n_kernels = 50;
one_svm_par.kernel = linspace(0.05,1.5,one_svm_par.n_kernels)';

% MULTI SVM
multi_svm_par.n_thresholds = 50;
multi_svm_par.threshold = linspace(0.5,1.0,multi_svm_par.n_thresholds)';
multi_svm_par.kernel_type = 'gauss';
multi_svm_par.n_kernels = 20;
multi_svm_par.kernel = linspace(1,4,multi_svm_par.n_kernels)';

% KPCA
kpca_par.n_thresholds = 50;
kpca_par.threshold = linspace(0.5,1.0,kpca_par.n_thresholds)';
kpca_par.kernel_type = 'gauss';
kpca_par.n_kernels = 20;
kpca_par.kernel = linspace(0.01,0.2,kpca_par.n_kernels)';

parameters = {knn_par,lmnn_par,klmnn_par,knfst_par,one_svm_par,multi_svm_par,kpca_par};

%% EXPERIMENTOS
% Roda experimentos de detec��o de novidade
%Methods.runExperiments(X,y,methods([4,5,6,7]),parameters,out_dir,n_experiments,...
%                        n_classes,untrained_classes,training_ratio,K,kappa,view_plot_metric);
                    
% Roda experimentos de detec��o de novidade para KNN, LMNN e KLMNN
%Methods.runExperimentsForKnnMethods(X,y,methods([1,2,3]),parameters,out_dir,...
%                   n_experiments,n_classes,untrained_classes,training_ratio,view_plot_metric);

%% CARREGA RESULTADOS
% Resultados para os m�todos baseados em KNN
Methods.reportExperimentsForKnnMethods(out_dir);

% Carrega e processa os resultados dos experimentos de detec��o de novidade
%Methods.reportExperimentResults(methods([1,2,3,4,5,6,7]),out_dir,K,kappa);
