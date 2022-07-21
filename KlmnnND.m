classdef KlmnnND < handle
  properties
    X = [];                 % Pontos X [n_points x dim]
    Y = [];                 % Rótulos Y
    n_classes = 0;          % Números de classes
    untrained_classes = 0;  % Número de classes não treinadas
    knn_arg = 0;            % O mesmo que o parâmetro K do artigo
    knn_threshold = 0;      % O mesmo que o parâmetroe "kappa" do artigo
    n_thresholds = 0;       % Número de threshold "tau" para validação
    threshold = [];         % Vetor de thresholds "tau" (o melhor deve ser encontrado)
    kernel_type = [];       % Tipo da função de kernel
    n_kernels = 0;          % Número de kernels para validação
    kernel = [];            % Vetor kernels para o kernel pca (o melhor deve ser encontrado)
    training_ratio = 0;     % Taxa de treinamento de amostras
    split = {};             % Guarda um objeto split para auxiliar o processo de validação
    samples_per_classe = 0; % Amostras por classe
  end
  
  methods
    % Construtor
    function obj = KlmnnND(X,Y,knn_arg,knn_threshold,n_classes,untrained_classes,training_ratio)
      obj.X = X;
      obj.Y = Y;
      obj.knn_arg = knn_arg;
      obj.knn_threshold = knn_threshold;
      obj.n_classes = n_classes;
      obj.training_ratio = 0.7;
      if nargin>=6
        obj.untrained_classes = untrained_classes;
        if nargin==7
          obj.training_ratio = training_ratio;
        end
      end
      % Código incluído ---------------------------------------------
      obj.samples_per_classe = sum(Y==unique(Y)',1);
      [obj.samples_per_classe,id] = sort(obj.samples_per_classe,'descend');
      obj.samples_per_classe = cat(1,id,obj.samples_per_classe);
      % -------------------------------------------------------------
    end
    
    % Executa experimentos de detecção de novidade e busca de hiperparâmetros
    function experiment = runNoveltyDetectionExperiments(obj,n_experiments,view_plot_metric)
      split_exp = cell(n_experiments,1);
      
      MCC = zeros(obj.n_kernels,obj.n_thresholds,n_experiments);
      AFR = zeros(obj.n_kernels,obj.n_thresholds,n_experiments);
      F1 = zeros(obj.n_kernels,obj.n_thresholds,n_experiments);
      TPR = zeros(obj.n_kernels,obj.n_thresholds,n_experiments);
      TNR = zeros(obj.n_kernels,obj.n_thresholds,n_experiments);
      FPR = zeros(obj.n_kernels,obj.n_thresholds,n_experiments);
      FNR = zeros(obj.n_kernels,obj.n_thresholds,n_experiments);
      
      evaluations = cell(obj.n_kernels,obj.n_thresholds,n_experiments);
      
      for i=1:n_experiments
        rng(i);
        % Seleciona classes treinadas e não treinadas
        [trained,untrained,is_trained_class] = Split.selectClasses(obj.n_classes,obj.untrained_classes);
        
        % Divide os índices em treino e teste
        [idx_train,idx_test] = Split.trainTestIdx(obj.X,obj.Y,obj.training_ratio,obj.n_classes,is_trained_class);
        [xtrain,xtest,ytrain,ytest] = Split.dataTrainTest(idx_train,idx_test,obj.X,obj.Y);
        
        % Todas as amostras de classes não treinadas são definidas
        % como outliers (label -1)
        ytest(logical(sum(ytest==untrained,2))) = -1;
        
        RK = [];
        for j=1:obj.n_kernels
          kernel_arg = obj.kernel(j);
          
          % Pré-processamento para o KPCA
          % treino
          mean_train = mean(xtrain);
          xtrain = xtrain - mean_train;
          max_train = max(xtrain(:));
          xtrain = xtrain/max_train;
          % teste
          xtest = xtest - mean_train;
          xtest = xtest/max_train;
          
          % KPCA
          fprintf('Compute KPCA... ');
          kpca = obj.kpcaModel(kernel_arg);
          xtrainp = kpca.train(xtrain); % Visualization.map(xtrain,xtrainp,ytrain)
          xtestp = kpca.test(xtest);
          fprintf('feito!\n');
          
          % Pré-processamento para o LMNN
          % treino
          mean_trainp = mean(xtrainp);
          xtrainp = xtrainp - mean_trainp;
          max_trainp = max(xtrainp(:));
          xtrainp = xtrainp/max_trainp;
          % teste
          xtestp = xtestp - mean_trainp;
          xtestp = xtestp/max_trainp;
          
          % LMNN
          lmnn = LmnnNovDetection(xtrainp,ytrain,obj.knn_arg,obj.knn_threshold,obj.n_classes,obj.untrained_classes);
          T = lmnn.computeTransform(xtrainp,ytrain);
          xtrainpg = lmnn.transform(xtrainp,T);
          xtestpg = lmnn.transform(xtestp,T);
          
          % KNN
          knn = KnnNovDetection(xtrainpg,ytrain,obj.knn_arg,obj.knn_threshold,obj.n_classes,obj.untrained_classes);
          RT = [];
          for k=1:obj.n_thresholds
            fprintf('\nKLMNN (K=%d kappa=%d) \tTest: %d/%d \tKernel (%d/%d) \tThreshold (%d/%d)\n',obj.knn_arg,obj.knn_threshold,i,n_experiments,j,obj.n_kernels,k,obj.n_thresholds);
            evaluations{j,k,i} = knn.evaluate(xtrainpg,ytrain,xtestpg,ytest,obj.threshold(k));
            evaluations{j,k,i}.kernel = kernel_arg;
            evaluations{j,k,i}.kpca_model = kpca;
            MCC(j,k,i) = evaluations{j,k,i}.MCC;
            F1(j,k,i) = evaluations{j,k,i}.F1;
            AFR(j,k,i) = evaluations{j,k,i}.AFR;
            TPR(j,k,i) = evaluations{j,k,i}.TPR;
            TNR(j,k,i) = evaluations{j,k,i}.TNR;
            FPR(j,k,i) = evaluations{j,k,i}.FPR;
            FNR(j,k,i) = evaluations{j,k,i}.FNR;
            if view_plot_metric
              RT = cat(1,RT,MCC(j,k,i));
              figure(1);
              clf('reset');
              plot(obj.threshold(1:k),RT,'-r','LineWidth',3);
              xlim([obj.threshold(1),obj.threshold(end)]);
              ylim([0,1]);
              xlabel('Threshold');
              ylabel('Matthews correlation coefficient (MCC)');
              title(['KLMNN [ test ',num2str(i),'/',num2str(n_experiments),' | kernel ',num2str(j),'/',num2str(obj.n_kernels),' | threshold ',num2str(k),'/',num2str(obj.n_thresholds),' ]']);
              drawnow;
              pause(0.0001);
            end
          end
          if view_plot_metric
            RK = cat(1,RK,max(RT));
            figure(2);
            clf('reset');
            plot(obj.kernel(1:j),RK,'-','LineWidth',3);
            xlim([obj.kernel(1),obj.kernel(end)]);
            ylim([0,1]);
            xlabel('Kernel');
            ylabel('Matthews correlation coefficient (MCC)');
            title(['KLMNN [ test ',num2str(i),'/',num2str(n_experiments),' | kernel ',num2str(j),'/',num2str(obj.n_kernels),' ]']);
            drawnow;
          end
        end
        split_exp{i}.trained_classes = trained;
        split_exp{i}.untrained_classes = untrained;
        split_exp{i}.idx_train = idx_train;
        split_exp{i}.idx_test = idx_test;
        split_exp{i}.xtrain = xtrain;
        split_exp{i}.xtest = xtest;
        split_exp{i}.ytrain = ytrain;
        split_exp{i}.ytest = ytest;
      end
      close all;
      % Métrica MCC
      mean_mcc = mean(MCC,3);
      max_mean_mcc = max(max(mean_mcc));
      [best_kernel_id,best_threshold_id] = find(mean_mcc == max_mean_mcc);
      best_kernel_id = best_kernel_id(1);
      best_threshold_id = best_threshold_id(1);
      
      % Demais métricas
      mean_f1 = mean(F1,3);
      mean_afr = mean(AFR,3);
      mean_tpr = mean(TPR,3);
      mean_tnr = mean(TNR,3);
      mean_fpr = mean(FPR,3);
      mean_fnr = mean(FNR,3);
      
      all_metrics.MCC = MCC;
      all_metrics.F1 = F1;
      all_metrics.AFR = AFR;
      all_metrics.TPR = TPR;
      all_metrics.TNR = TNR;
      all_metrics.FPR = FPR;
      all_metrics.FNR = FNR;
      experiment.all_metrics = all_metrics;
      
      model.training_ratio = obj.training_ratio;
      model.best_threshold_id = best_threshold_id;
      model.best_kernel_id = best_kernel_id;
      model.threshold = obj.threshold(best_threshold_id);
      model.kernel = obj.kernel(best_kernel_id);
      model.untrained_classes = obj.untrained_classes;
      model.knn_arg = obj.knn_arg;
      model.knn_threshold = obj.knn_threshold;
      
      experiment.model = model;
      experiment.split = cell2mat(split_exp);
      experiment.evaluations = evaluations;
      experiment.mean_mcc = mean_mcc;
      experiment.mean_f1 = mean_f1;
      experiment.mean_afr = mean_afr;
      experiment.mean_tpr = mean_tpr;
      experiment.mean_tnr = mean_tnr;
      experiment.mean_fpr = mean_fpr;
      experiment.mean_fnr = mean_fnr;
      
      experiment.mcc_score = mean_mcc(best_kernel_id,best_threshold_id);
      experiment.f1_score = mean_f1(best_kernel_id,best_threshold_id);
      experiment.afr_score = mean_afr(best_kernel_id,best_threshold_id);
      experiment.tpr_score = mean_tpr(best_kernel_id,best_threshold_id);
      experiment.tnr_score = mean_tnr(best_kernel_id,best_threshold_id);
      experiment.fpr_score = mean_fpr(best_kernel_id,best_threshold_id);
      experiment.fnr_score = mean_fnr(best_kernel_id,best_threshold_id);
      experiment.all_metrics = all_metrics;
      
      fprintf('\nRESULTS\n MCC Score: %.4f\n F1 Score: %.4f\n AFR Score: %.4f\n',...
        experiment.mcc_score,experiment.f1_score,experiment.afr_score);
      
      figure; pcolor(obj.threshold,obj.kernel,mean_mcc); colorbar;
      xlabel('threshold'); ylabel('kernel');  title('MCC');
      
      figure; pcolor(obj.threshold,obj.kernel,mean_afr); colorbar;
      xlabel('threshold'); ylabel('kernel'); title('AFR');
    end
    
    % Validação do algoritmo klmnn out detection
    function model = validation(obj,n_validations,view_plot_error)
      obj.split = cell(n_validations,1);
      mcc = zeros(obj.n_kernels,obj.n_thresholds,n_validations);
      for i=1:n_validations
        rng(i);
        % Cria um objeto split. Particiona a base em dois conjuntos
        % de classes treinadas e não treinadas. Separa uma
        % parte para treinamento e outra para teste
        obj.split{i} = SplitData(obj.X,obj.Y,obj.training_ratio,obj.untrained_classes);
        % Separa uma parte do treinamento para validação
        [id_train,id_val] = obj.split{i}.idTrainVal();
        [xtrain,ytrain,xval,yval] = obj.split{i}.dataTrainVal(id_train,id_val);
        RK = [];
        for j=1:obj.n_kernels
          kernel_arg = obj.kernel(j);
          
          % Pré-processamento para o KPCA
          % treino
          mean_train = mean(xtrain);
          xtrain = xtrain - mean_train;
          max_train = max(xtrain(:));
          xtrain = xtrain/max_train;
          % teste
          xval = xval - mean_train;
          xval = xval/max_train;
          
          % KPCA
          fprintf('Compute KPCA... ');
          kpca = obj.kpcaModel(kernel_arg);
          xtrainp = kpca.train(xtrain); % Visualization.map(xtrain,xtrainp,ytrain)
          xvalp = kpca.test(xval);
          fprintf('feito!\n');
          
          % Pré-processamento para o LMNN
          % treino
          mean_trainp = mean(xtrainp);
          xtrainp = xtrainp - mean_trainp;
          max_trainp = max(xtrainp(:));
          xtrainp = xtrainp/max_trainp;
          % teste
          xvalp = xvalp - mean_trainp;
          xvalp = xvalp/max_trainp;
          
          % LMNN
          lmnn = LmnnNovDetection(xtrainp,ytrain,obj.knn_arg,obj.knn_threshold,obj.n_classes,obj.untrained_classes);
          T = lmnn.computeTransform(xtrainp,ytrain);
          xtrainpg = lmnn.transform(xtrainp,T);
          xvalpg = lmnn.transform(xvalp,T);
          
          % KNN
          knn = KnnNovDetection(xtrainpg,ytrain,obj.knn_arg,obj.knn_threshold,obj.n_classes,obj.untrained_classes);
          RT = [];
          for k=1:obj.n_thresholds
            fprintf('\nKLMNN (K=%d kappa=%d) \tVal: %d/%d \tKernel (%d/%d) \tThreshold (%d/%d)\n',obj.knn_arg,obj.knn_threshold,i,n_validations,j,obj.n_kernels,k,obj.n_thresholds);
            result = knn.evaluate(xtrainpg,ytrain,xvalpg,yval,obj.threshold(k));
            result.kernel = kernel_arg;
            mcc(j,k,i) = result.MCC;
            if view_plot_error
              RT = cat(1,RT,mcc(j,k,i));
              figure(1);
              clf('reset');
              plot(obj.threshold(1:k),RT,'-r','LineWidth',3);
              xlim([obj.threshold(1),obj.threshold(end)]);
              ylim([0,1]);
              xlabel('Threshold');
              ylabel('Matthews correlation coefficient (MCC)');
              title(['KLMNN [ validação ',num2str(i),'/',num2str(n_validations),' | kernel ',num2str(j),'/',num2str(obj.n_kernels),' | threshold ',num2str(k),'/',num2str(obj.n_thresholds),' ]']);
              drawnow;
              pause(0.01);
            end
          end
          if view_plot_error
            RK = cat(1,RK,max(RT));
            figure(2);
            clf('reset');
            plot(obj.kernel(1:j),RK,'-','LineWidth',3);
            xlim([obj.kernel(1),obj.kernel(end)]);
            ylim([0,1]);
            xlabel('Kernel');
            ylabel('Matthews correlation coefficient (MCC)');
            title(['KLMNN [ validação ',num2str(i),'/',num2str(n_validations),' | kernel ',num2str(j),'/',num2str(obj.n_kernels),' ]']);
            drawnow;
          end
        end
        model.split{i} = obj.split{i};
      end
      close all;
      mean_mcc = mean(mcc,3);
      max_mean_mcc = max(max(mean_mcc));
      [id_k,id_t] = find(mean_mcc == max_mean_mcc);
      id_k = id_k(1); id_t = id_t(1);
      
      model.training_ratio = obj.training_ratio;
      model.kernel = obj.kernel(id_k);
      model.threshold = obj.threshold(id_t);
      model.untrained_classes = obj.untrained_classes;
      model.knn_arg = obj.knn_arg;
      model.knn_threshold = obj.knn_threshold;
      model.mean_mcc = max_mean_mcc;
    end
    
    % Avalia o modelo treinado
    function [results,evaluations] = evaluateModel(obj,model,n_tests)
      evaluations = cell(n_tests,1);
      for i=1:n_tests
        rng(i);
        fprintf('\nKLMNN (K=%d kappa=%d) \tTest: %d/%d\n',obj.knn_arg,obj.knn_threshold,i,n_tests);
        id_test = obj.split{i}.idTest();
        [xtest,ytest] = obj.split{i}.dataTest(id_test);
        [xtrain,ytrain] = obj.split{i}.dataTrain(obj.split{i}.id_train_val_t);
        evaluations{i} = obj.evaluate(xtrain,ytrain,xtest,ytest,model.kernel,model.threshold);
        evaluations{i}.kernel = model.kernel;
      end
      results = struct2table(cell2mat(evaluations));
    end
    
    % Avalia o modelo treinado em conjuntos de testes
    function [results,evaluations] = evaluateTests(obj,xtrain,ytrain,xtest,ytest,model)
      n_tests = size(xtest,3);
      evaluations = cell(n_tests,1);
      for i=1:n_tests
        fprintf('\nKLMNN (K=%d kappa=%d) \tTest: %d/%d\n',obj.knn_arg,obj.knn_threshold,i,n_tests);
        evaluations{i} = obj.evaluate(xtrain,ytrain,xtest(:,:,i),ytest,model.kernel,model.threshold);
        evaluations{i}.kernel = model.kernel;
      end
      results = struct2table(cell2mat(evaluations));
    end
    
    % Avalia o klmnn out detection
    function result = evaluate(obj,xtrain,ytrain,xtest,ytest,kernel_arg,threshold)
      % Pré-processamento para o KPCA
      % treino
      mean_train = mean(xtrain);
      xtrain = xtrain - mean_train;
      max_train = max(xtrain(:));
      xtrain = xtrain/max_train;
      % teste
      xtest = xtest - mean_train;
      xtest = xtest/max_train;
      
      % KPCA
      kpca = obj.kpcaModel(kernel_arg);
      xtrainp = kpca.train(xtrain);
      xtestp = kpca.test(xtest);
      
      % Pré-processamento para o LMNN
      % treino
      mean_trainp = mean(xtrainp);
      xtrainp = xtrainp - mean_trainp;
      max_trainp = max(xtrainp(:));
      xtrainp = xtrainp/max_trainp;
      % teste
      xtestp = xtestp - mean_trainp;
      xtestp = xtestp/max_trainp;
      
      % LMNN
      lmnn = LmnnNovDetection(xtrainp,ytrain,obj.knn_arg,obj.knn_threshold,obj.n_classes,obj.untrained_classes);
      T = lmnn.computeTransform(xtrainp,ytrain);
      xtrainpg = lmnn.transform(xtrainp,T);
      xtestpg = lmnn.transform(xtestp,T);
      
      % KNN
      knn = KnnNovDetection(xtrainpg,ytrain,obj.knn_arg,obj.knn_threshold,obj.n_classes,obj.untrained_classes);
      result = knn.evaluate(xtrainpg,ytrain,xtestpg,ytest,threshold);
      result.kpca_model = kpca;
    end
    
    % Avalia o klmnn out detection
    function predictions = predict(obj,xtrain,ytrain,xtest,kernel_arg,threshold)
      % Pré-processamento para o KPCA
      % treino
      mean_train = mean(xtrain);
      xtrain = xtrain - mean_train;
      max_train = max(xtrain(:));
      xtrain = xtrain/max_train;
      % teste
      xtest = xtest - mean_train;
      xtest = xtest/max_train;
      
      % KPCA
      kpca = obj.kpcaModel(kernel_arg);
      xtrainp = kpca.train(xtrain);
      xtestp = kpca.test(xtest);
      
      % Pré-processamento para o LMNN
      % treino
      mean_trainp = mean(xtrainp);
      xtrainp = xtrainp - mean_trainp;
      max_trainp = max(xtrainp(:));
      xtrainp = xtrainp/max_trainp;
      % teste
      xtestp = xtestp - mean_trainp;
      xtestp = xtestp/max_trainp;
      
      % LMNN
      lmnn = LmnnNovDetection(xtrainp,ytrain,obj.knn_arg,obj.knn_threshold,obj.n_classes,obj.untrained_classes);
      T = lmnn.computeTransform(xtrainp,ytrain);
      xtrainpg = lmnn.transform(xtrainp,T);
      xtestpg = lmnn.transform(xtestp,T);
      
      % KNN
      knn = KnnNovDetection(xtrainpg,ytrain,obj.knn_arg,obj.knn_threshold,obj.n_classes,obj.untrained_classes);
      predictions = knn.predict(xtrainpg,ytrain,xtestpg,threshold);
    end
    
    % Calcula o kpca
    function model = kpcaModel(obj,kernel_arg)
      % set kernel function
      if strcmp(obj.kernel_type,'poly')
        kernel_f = Kernel('type','poly','degree',kernel_arg,'offset',0.5);
        % parameter setting
        parameter = struct('application', 'dr','display','on',...
          'kernel',kernel_f,'tol',1e-6);
      else
        kernel_f = Kernel('type','gauss','width',kernel_arg);
        % parameter setting
        parameter = struct('application','dr','display','on',...
          'kernel',kernel_f,'explained',0.9,'tol',1e-6);
      end
      
      % build a KPCA object
      model = KernelPCA(parameter);
    end
  end
end


