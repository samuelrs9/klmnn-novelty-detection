%% Inicializações
clc; clear; close all;
run('libraries/lmnn/setpaths3.m');
addpath('libraries/kpca');
addpath('libraries/knfst');
addpath(pwd);

%% CARREGA A BASE
libras = Bases.loadGlass;
X = libras.X; 
y = libras.Y;

out_dir = 'glass';
n_classes = numel(unique(y));

%% CONFIGURAÇÕES
methods = {'knn','lmnn','klmnn','knfst','one_svm','multi_svm','kpca'};

n_experiments = 10;
untrained_classes = floor(0.5*n_classes);
training_ratio = 0.8;
view_plot_metric = false;

% HIPERPARÂMETROS
% Para KNN, LMNN e KLMNN
K = 1;
kappa = 1;

% Define intervalos de busca de parâmetros de kernel e thresholds
% KNN
knn_par.n_thresholds = 50;
knn_par.threshold = linspace(0.1,1.1,knn_par.n_thresholds)';

% LMNN
lmnn_par.n_thresholds = 50;
lmnn_par.threshold = linspace(0.1,1.5,lmnn_par.n_thresholds)';

% KLMNN
klmnn_par.n_thresholds = 50;
klmnn_par.threshold = linspace(0.1,1.5,klmnn_par.n_thresholds)';
klmnn_par.n_kernels = 20;
klmnn_par.kernel_type = 'gauss';
klmnn_par.kernel = linspace(0.5,1.5,klmnn_par.n_kernels)';

% KNFST
knfst_par.n_thresholds = 50;
knfst_par.threshold = linspace(0.22,0.35,knfst_par.n_thresholds)';
knfst_par.kernel_type = 'gauss';
knfst_par.n_kernels = 20;
knfst_par.kernel = linspace(0.15,0.25,knfst_par.n_kernels)';

% ONE SVM
one_svm_par.kernel_type = 'gauss';
one_svm_par.n_kernels = 50;
one_svm_par.kernel = linspace(1.2,1.5,one_svm_par.n_kernels)';

% MULTI SVM
multi_svm_par.n_thresholds = 50;
multi_svm_par.threshold = linspace(0.6,1.0,multi_svm_par.n_thresholds)';
multi_svm_par.kernel_type = 'gauss';
multi_svm_par.n_kernels = 20;
multi_svm_par.kernel = linspace(2.5,4.0,multi_svm_par.n_kernels)';

% KPCA
kpca_par.n_thresholds = 50;
kpca_par.threshold = linspace(0.5,1.0,kpca_par.n_thresholds)';
kpca_par.kernel_type = 'gauss';
kpca_par.n_kernels = 20;
kpca_par.kernel = linspace(0.01,0.6,kpca_par.n_kernels)';

parameters = {knn_par,lmnn_par,klmnn_par,knfst_par,one_svm_par,multi_svm_par,kpca_par};

%% EXPERIMENTOS
% Roda experimentos de detecção de novidade
%Methods.runExperiments(X,y,methods([4,5,6,7]),parameters,out_dir,n_experiments,...
%                        n_classes,untrained_classes,training_ratio,K,kappa,view_plot_metric);
                    
% Roda experimentos de detecção de novidade para KNN, LMNN e KLMNN
%Methods.runExperimentsForKnnMethods(X,y,methods([1,2,3]),parameters,out_dir,...
%                   n_experiments,n_classes,untrained_classes,training_ratio,view_plot_metric);

%% CARREGA RESULTADOS
% Resultados para os métodos baseados em KNN
%Methods.reportExperimentsForKnnMethods(out_dir);

% Carrega e processa os resultados dos experimentos de detecção de novidade
Methods.reportExperimentResults(methods([1,2,3,4,5,6,7]),out_dir,K,kappa);
