%% INICIALIZAÇÕES
clc; clear; close all;
run('libraries/lmnn/setpaths3.m');
addpath('libraries/kpca');
addpath('libraries/knfst');

%% CARREGA A BASE
out_dir = 'simulation_3';
load(strcat(out_dir,'/base.mat'));

n_classes = numel(unique(y));

%% CONFIGURAÇÕES
methods = {'knn','lmnn','klmnn','knfst','one_svm','multi_svm','kpca'};

search_parameters = true;

if search_parameters
    training_ratio = 0.7;
    n_experiments = 10;
    untrained_classes = 1;
    
    view_plot_metric = true;
    
    % HIPERPARÂMETROS
    % Para KNN, LMNN e KLMNN
    K = 2;
    kappa = 1;

    % Define intervalos de busca de parâmetros de kernel e thresholds
    % KNN
    knn_par.n_thresholds = 20;
    knn_par.threshold = linspace(0.01,1.0,knn_par.n_thresholds)';

    % LMNN
    lmnn_par.n_thresholds = 20;
    lmnn_par.threshold = linspace(0.01,1.0,lmnn_par.n_thresholds)';

    % KLMNN
    klmnn_par.n_thresholds = 20;
    klmnn_par.threshold = linspace(0.01,3.0,klmnn_par.n_thresholds)';
    klmnn_par.n_kernels = 5;
    klmnn_par.kernel_type = 'poly';
    klmnn_par.kernel = linspace(1,5,klmnn_par.n_kernels)';

    % KNFST
    knfst_par.n_thresholds = 20;
    knfst_par.threshold = linspace(0.001,0.1,knfst_par.n_thresholds)';
    knfst_par.kernel_type = 'poly';
    knfst_par.n_kernels = 5;
    knfst_par.kernel = linspace(1,5,knfst_par.n_kernels)';

    % ONE SVM
    one_svm_par.kernel_type = 'gauss';
    one_svm_par.n_kernels = 50;
    one_svm_par.kernel = linspace(0.1,2.0,one_svm_par.n_kernels)';

    % MULTI SVM
    multi_svm_par.n_thresholds = 20;
    multi_svm_par.threshold = linspace(0.3,1.0,multi_svm_par.n_thresholds)';
    multi_svm_par.kernel_type = 'gauss';
    multi_svm_par.n_kernels = 10;
    multi_svm_par.kernel = linspace(3,6,multi_svm_par.n_kernels)';

    % KPCA
    kpca_par.n_thresholds = 20;    
    kpca_par.threshold = linspace(0.5,1.0,kpca_par.n_thresholds)';
    kpca_par.kernel_type = 'poly';
    kpca_par.n_kernels = 10;
    kpca_par.kernel = linspace(1,5,kpca_par.n_kernels)';

    parameters = {knn_par,lmnn_par,klmnn_par,knfst_par,one_svm_par,multi_svm_par,kpca_par};

    %% VALIDAÇÃO DOS HIPERPARÂMETROS
    % Validação dos métodos      
    %Methods.runValidations(X,y,methods([1]),parameters,out_dir,n_experiments,...
    %                       n_classes,untrained_classes,training_ratio,K,kappa,view_plot_metric);
    
    % Validação dos hiperparâmetros K e kappa para os métodos knn, lmnn e klmnn
    % Methods.runValidationsForKnnMethods(X,y,methods([1,2]),parameters,out_dir,n_experiments,...
    %                       n_classes,untrained_classes,training_ratio,view_plot_metric)
    
    %% AVALIAÇÃO
    % Avalia os métodos e salva as métricas de acurácia
    Methods.runEvaluations(xtrain,ytrain,xtest,ytest,methods([5]),out_dir,n_classes,K,kappa);
    
    % Avalia os métodos knn, lmnn, klmnn e salva as métricas de acurácia
    %Methods.runEvaluationsForKnnMethods(xtrain,ytrain,xtest,ytest,methods([1,2]),out_dir,n_experiments);
        
    % Executa a predição dos métodos em pontos do grid e desenha as fronteiras das classes
    %Methods.runPredictions(X,y,xtest,methods([1]),out_dir,n_classes,K,kappa);    
    
    %% CARREGA RESULTADOS
    % Resultado dos métodos
    Methods.reportEvaluation(methods([1,2,3,4,5,6,7]),out_dir,K,kappa);
    
    % Resultados da variação dos parâmetros K e kappa
    %Methods.reportEvaluationForKnnMethods(out_dir);
    
else
    % TESTA COM PARÂMETROS MANUAIS
    % Para KNN, LMNN e KLMNN
    K = 2;
    kappa = 1;
   
    % KNN
    knn_par.threshold_arg = 0.1;
    
    % LMNN
    lmnn_par.threshold_arg = 1;
    
    % KLMNN
    klmnn_par.threshold_arg = 1.0;
    klmnn_par.kernel_type = 'poly'; % ou 'poly';
    klmnn_par.kernel_arg = 2;

    % KNFST
    knfst_par.threshold_arg = 0.1;
    knfst_par.kernel_type = 'poly';
    knfst_par.kernel_arg = 2;

    % ONE SVM
    one_svm_par.kernel_type = 'gauss';
    one_svm_par.kernel_arg = 0.5;

    % MULTI SVM
    multi_svm_par.threshold_arg = 0.5;
    multi_svm_par.kernel_type = 'gauss';    
    multi_svm_par.kernel_arg = 0.5;

    % KPCA NOV   
    kpca_par.threshold_arg = 1.0;
    kpca_par.kernel_type = 'gauss';
    kpca_par.kernel_arg = 0.8;

    parameters = {knn_par,lmnn_par,klmnn_par,knfst_par,one_svm_par,multi_svm_par,kpca_par};
    
    % Executa a predição dos métodos em pontos do grid e desenha as fronteiras das classes
    Methods.runPredictionsParameter(xtrain,ytrain,xtest,methods([7]),parameters,out_dir,n_classes,K,kappa);
end
                        