%% INICIALIZAÇÕES
clc; clear; close all;
run('libraries/lmnn/setpaths3.m');
addpath('libraries/kpca');
addpath('libraries/knfst');

out_dir = 'simulation_4';

% VARIAÇÃO DA QUANTIDADE DE DADOS DE TREINAMENTO E DIMENSÕES
N = [100,200,300,400,500,750,1000,1500,2000];
%N = 800;

% MÉTODOS
methods = {'knn','lmnn','klmnn','knfst','one_svm','multi_svm','kpca'};

% PARA KNN, LMNN e KLMNN
K = 3;
kappa = 1;

run_validation = false;

if run_validation
    for i=1:numel(N)
        if N(i) == 800
            DIM = 2:20;
        else
            DIM = 10;
        end
        for j=1:numel(DIM)
            % CRIA A BASE      
            base = Simulations.uniformDistributions(N(i),DIM(j));
            X = base.X;
            y = base.y;
            xtrain = base.xtrain;
            ytrain = base.ytrain;
            xtest = base.xtest;
            ytest = base.ytest;

            exp_dir = strcat(out_dir,'/N=',int2str(N(i)),' DIM=',int2str(DIM(j)));

            training_ratio = 1.0;
            n_classes = numel(unique(y));
            max_std = max(std(X));

            search_parameters = true;        

            if search_parameters
                % CALIBRAÇÃO DOS PARÂMETROS POR FORÇA BRUTA
                n_experiments = 5;
                untrained_classes = 1;

                view_plot_metric = false;

                % Para KNN, LMNN e KLMNN
                K = 3;
                kappa = 1;

                % Define intervalos de busca de parâmetros de kernel e thresholds
                % KNN
                knn_par.n_thresholds = 20;
                knn_par.threshold = linspace(0.5,1.5,knn_par.n_thresholds)';

                % LMNN
                lmnn_par.n_thresholds = 20;
                lmnn_par.threshold = linspace(0.5,1.5,lmnn_par.n_thresholds)';

                % KLMNN
                klmnn_par.n_thresholds = 20;
                klmnn_par.threshold = linspace(0.5,4.5,klmnn_par.n_thresholds)';
                klmnn_par.n_kernels = 10;
                klmnn_par.kernel_type = 'gauss'; % explained = 0.9
                klmnn_par.kernel = linspace(0.5*max_std,3.0*max_std,klmnn_par.n_kernels)';

                % KNFST
                knfst_par.n_thresholds = 20;
                knfst_par.threshold = linspace(0.01,0.7,knfst_par.n_thresholds)';
                knfst_par.kernel_type = 'gauss';
                knfst_par.n_kernels = 10;
                knfst_par.kernel = linspace(0.01*max_std,1.5*max_std,knfst_par.n_kernels)';

                % ONE SVM
                one_svm_par.kernel_type = 'gauss';
                one_svm_par.n_kernels = 20;
                one_svm_par.kernel = linspace(0.01*max_std,1.5*max_std,one_svm_par.n_kernels)';

                % MULTI SVM
                multi_svm_par.n_thresholds = 20;
                multi_svm_par.threshold = linspace(0.0,1.0,multi_svm_par.n_thresholds)';
                multi_svm_par.kernel_type = 'gauss';
                multi_svm_par.n_kernels = 10;
                multi_svm_par.kernel = linspace(0.01*max_std,1.5*max_std,multi_svm_par.n_kernels)';

                % KPCA
                kpca_par.n_thresholds = 20;
                kpca_par.threshold = linspace(0.5,0.99,kpca_par.n_thresholds)';
                kpca_par.kernel_type = 'gauss';
                kpca_par.n_kernels = 10;
                kpca_par.kernel = linspace(0.01*max_std,1.5*max_std,kpca_par.n_kernels)';

                parameters = {knn_par,lmnn_par,klmnn_par,knfst_par,one_svm_par,multi_svm_par,kpca_par};

                % Validação dos métodos
                %Methods.runValidations(X,y,methods([1,2,3,4,5,6,7]),parameters,exp_dir,n_experiments,...
                %                       n_classes,untrained_classes,training_ratio,K,kappa,view_plot_metric);

                % Avaliação dos métodos (Usa os 30% dos dados que não foram utilizados na validação)
                %Methods.runEvaluationModels(methods([1,2,3,4,5,6,7]),out_dir,n_experiments,K,kappa);

                % Avalia os métodos conjuntos de teste e salva as métricas de acurácia
                %Methods.runEvaluationTests(xtrain,ytrain,xtest,ytest,methods([1,2,3,4,5,6,7]),exp_dir,n_classes,K,kappa);
            else
                % TESTA COM PARÂMETROS MANUAIS
                % KNN
                knn_par.threshold_arg = 1;

                % LMNN
                lmnn_par.threshold_arg = 1;

                % KLMNN
                klmnn_par.threshold_arg = 1.0;    
                klmnn_par.kernel_type = 'gauss'; % ou 'poly';
                klmnn_par.kernel_arg = 0.5;

                % KNFST
                knfst_par.threshold_arg = 1;
                knfst_par.kernel_type = 'gauss';
                knfst_par.kernel_arg = 0.5;

                % ONE SVM
                one_svm_par.kernel_type = 'gauss';
                one_svm_par.kernel_arg = 0.5;

                % MULTI SVM
                multi_svm_par.threshold_arg = 0.5;
                multi_svm_par.kernel_type = 'gauss';    
                multi_svm_par.kernel_arg = 0.5;

                % KPCA NOV   
                kpca_par.threshold_arg = 0.2;
                kpca_par.kernel_type = 'gauss';
                kpca_par.kernel_arg = 0.5;

                parameters = {knn_par,lmnn_par,klmnn_par,knfst_par,one_svm_par,multi_svm_par,kpca_par};

                % Avalia os métodos
                Methods.runEvaluationsParameter(xtrain,ytrain,X,y,methods([1,2,3,4,5,6,7]),parameters,exp_dir,n_classes,K,kappa);
            end
        end
    end
end

DIM = 2:20;

% Carrega e processa o experimento de tempo de execução e avalia métricas
Methods.reportExecutionTimeAndMetricsTests(methods,out_dir,N,DIM,K,kappa)
