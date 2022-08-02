classdef KnfstND < handle
  % --------------------------------------------------------------------------------------
  % KNFST Novelty Detection for multi-class classification problems.
  %
  % Version 2.0, July 2022.
  % By Samuel Silva (samuelrs@usp.br).
  % --------------------------------------------------------------------------------------  
  properties
    X = [];                  % data samples [num_samples x dimension]
    Y = [];                  % labels [num_samples x 1]
    num_samples = 0;         % number of samples in dataset
    dimension = 0;           % data dimension
    num_classes = 0;         % number of classes
    untrained_classes = 0;   % number of untrained classes
    num_thresholds = 0;      % number of score thresholds
    threshold = [];          % score thresholds list (the best needs to be found)
    training_ratio = 0;      % training sample rate
    split = {};              % holds a split object that helps the cross-validation process
    samples_per_classe = []; % samples per class
    kernel_type = [];        % kernel function type
    num_kernels = 0;         % number of kernel values
    kernel = [];             % kernel list for svm algorithm (the best must be found)    
  end
  
  methods
    function obj = KnfstND(X,Y,num_classes,untrained_classes,training_ratio)
      % ----------------------------------------------------------------------------------
      % Construtor.
      % ----------------------------------------------------------------------------------
      obj.X = X;
      obj.Y = Y;
      obj.num_classes = num_classes;
      obj.training_ratio = 0.7;
      if nargin>=4
        obj.untrained_classes = untrained_classes;
        if nargin==5
          obj.training_ratio = training_ratio;
        end
      end
    end

    function experiment = runExperiments(obj,num_experiments,plot_metric)
      % ----------------------------------------------------------------------------------
      % Executa experimentos de detecção de novidade e busca de hiperparâmetros
      % ----------------------------------------------------------------------------------      
      split_exp = cell(num_experiments,1);
      
      MCC = zeros(obj.num_kernels,obj.num_thresholds,num_experiments);
      AFR = zeros(obj.num_kernels,obj.num_thresholds,num_experiments);
      F1 = zeros(obj.num_kernels,obj.num_thresholds,num_experiments);
      TPR = zeros(obj.num_kernels,obj.num_thresholds,num_experiments);
      TNR = zeros(obj.num_kernels,obj.num_thresholds,num_experiments);
      FPR = zeros(obj.num_kernels,obj.num_thresholds,num_experiments);
      FNR = zeros(obj.num_kernels,obj.num_thresholds,num_experiments);
      
      evaluations = cell(obj.num_kernels,obj.num_thresholds,num_experiments);
      
      for i=1:num_experiments
        rng(i);
        % Seleciona classes treinadas e não treinadas
        [trained,untrained,is_trained_class] = Split.selectClasses(obj.num_classes,obj.untrained_classes);
        
        % Divide os índices em treino e teste
        [idx_train,idx_test] = Split.trainTestIdx(obj.X,obj.Y,obj.training_ratio,obj.num_classes,is_trained_class);
        [xtrain,xtest,ytrain,ytest] = Split.dataTrainTest(idx_train,idx_test,obj.X,obj.Y);
        
        % Todas as amostras não treinadas são definidas
        % como outliers (label -1)
        ytest(logical(sum(ytest==untrained,2))) = -1;
        
        RK = [];
        for j=1:obj.num_kernels
          kernel_arg = obj.kernel(j);
          
          % Matriz de Kernel Treinamento x Treinamento
          KTr = obj.kernelMatrix(xtrain,xtrain,kernel_arg);
          
          % Matriz de Kernel Treinamento x Teste
          KTe = obj.kernelMatrix(xtrain,xtest,kernel_arg);
          
          RT = [];
          for k=1:obj.num_thresholds
            fprintf('\nKNFST \tTest: %d/%d \tKernel (%d/%d) \tThreshold (%d/%d)\n',i,num_experiments,j,obj.num_kernels,k,obj.num_thresholds);
            threshold_arg = obj.threshold(k);
            evaluations{j,k,i} = obj.evaluateAux(KTr,ytrain,KTe,ytest,kernel_arg,threshold_arg);
            evaluations{j,k,i}.kernel = kernel_arg;
            MCC(j,k,i) = evaluations{j,k,i}.MCC;
            F1(j,k,i) = evaluations{j,k,i}.F1;
            AFR(j,k,i) = evaluations{j,k,i}.AFR;
            TPR(j,k,i) = evaluations{j,k,i}.TPR;
            TNR(j,k,i) = evaluations{j,k,i}.TNR;
            FPR(j,k,i) = evaluations{j,k,i}.FPR;
            FNR(j,k,i) = evaluations{j,k,i}.FNR;
            if plot_metric
              RT = cat(1,RT,MCC(j,k,i));
              figure(1);
              clf('reset');
              plot(obj.threshold(1:k),RT,'-r','LineWidth',3);
              xlim([obj.threshold(1),obj.threshold(end)]);
              ylim([0,1]);
              xlabel('Threshold');
              ylabel('Matthews correlation coefficient (MCC)');
              title(['KNFST [ test ',num2str(i),'/',num2str(num_experiments),' | kernel ',num2str(j),'/',num2str(obj.num_kernels),' | threshold ',num2str(k),'/',num2str(obj.num_thresholds),' ]']);
              drawnow;
              pause(0.01);
            end
          end
          if plot_metric
            RK = cat(1,RK,max(RT));
            figure(2);
            clf('reset');
            plot(obj.kernel(1:j),RK,'-','LineWidth',3);
            xlim([obj.kernel(1),obj.kernel(end)]);
            ylim([0,1]);
            xlabel('Kernel');
            ylabel('Matthews correlation coefficient (MCC)');
            title(['KNFST [ test ',num2str(i),'/',num2str(num_experiments),' | kernel ',num2str(j),'/',num2str(obj.num_kernels),' ]']);
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
      
      
      fprintf('\nRESULTS\n MCC Score: %.4f\n F1 Score: %.4f\n AFR Score: %.4f\n',...
        experiment.mcc_score,experiment.f1_score,experiment.afr_score);
      
      figure;  pcolor(obj.threshold,obj.kernel,mean_mcc); colorbar;
      xlabel('threshold'); ylabel('kernel'); title('MCC');
      
      figure; pcolor(obj.threshold,obj.kernel,mean_afr); colorbar;
      xlabel('threshold'); ylabel('kernel'); title('AFR');
    end
    
    function model = validation(obj,n_validations,view_plot_error)
      % ----------------------------------------------------------------------------------
      % Validação do algoritmo knfst
      % ----------------------------------------------------------------------------------      
      obj.split = cell(n_validations,1);
      mcc = zeros(obj.num_kernels,obj.num_thresholds,n_validations);
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
        for j=1:obj.num_kernels
          kernel_arg = obj.kernel(j);
          % Matriz de Kernel Treinamento x Treinamento
          KTr = obj.kernelMatrix(xtrain,xtrain,kernel_arg);
          % Matriz de Kernel Treinamento x Validação
          KVa = obj.kernelMatrix(xtrain,xval,kernel_arg);
          RT = [];
          for k=1:obj.num_thresholds
            %if rem(k,20) == 0
            fprintf('\nKNFST \tVal: %d/%d \tKernel %d/%d \tThreshold %d/%d\n',i,n_validations,j,obj.num_kernels,k,obj.num_thresholds);
            %end
            threshold_arg = obj.threshold(k);
            result = obj.evaluateAux(KTr,ytrain,KVa,yval,kernel_arg,threshold_arg);
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
              title(['KNFST [ validação ',num2str(i),'/',num2str(n_validations),' | kernel ',num2str(j),'/',num2str(obj.num_kernels),' | threshold ',num2str(k),'/',num2str(obj.num_thresholds),' ]']);
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
            title(['KNFST [ validação ',num2str(i),'/',num2str(n_validations),' | kernel ',num2str(j),'/',num2str(obj.num_kernels),' ]']);
            drawnow;
          end
        end
        model.split{i} = obj.split{i};
      end
      mean_mcc = mean(mcc,3);
      max_mean_mcc = max(max(mean_mcc));
      [id_k,id_t] = find(mean_mcc == max_mean_mcc);
      id_k = id_k(1); id_t = id_t(1);
      
      model.training_ratio = obj.training_ratio;
      model.kernel = obj.kernel(id_k);
      model.threshold = obj.threshold(id_t);
      model.untrained_classes = obj.untrained_classes;
      model.mean_mcc = max_mean_mcc;
    end
    
    function [results,evaluations] = evaluateModel(obj,model,n_tests)
      % ----------------------------------------------------------------------------------
      % Avalia o modelo treinado
      % ----------------------------------------------------------------------------------      
      evaluations = cell(n_tests,1);
      for i=1:n_tests
        rng(i);
        fprintf('\nKNFST Test: %d/%d\n',i,n_tests);
        id_test = obj.split{i}.idTest();
        [xtest,ytest] = obj.split{i}.dataTest(id_test);
        [xtrain,ytrain] = obj.split{i}.dataTrain(obj.split{i}.id_train_val_t);
        evaluations{i} = obj.evaluate(xtrain,ytrain,xtest,ytest,model.kernel,model.threshold);
      end
      results = struct2table(cell2mat(evaluations));
    end

    function [results,evaluations] = evaluateTests(obj,xtrain,ytrain,xtest,ytest,model)
      % ----------------------------------------------------------------------------------
      % Avalia o modelo treinado em conjuntos de testes
      % ----------------------------------------------------------------------------------      
      n_tests = size(xtest,3);
      evaluations = cell(n_tests,1);
      for i=1:n_tests
        fprintf('\nKNFST \tTest: %d/%d\n',i,n_tests);
        evaluations{i} = obj.evaluate(xtrain,ytrain,xtest(:,:,i),ytest,model.kernel,model.threshold);
      end
      results = struct2table(cell2mat(evaluations));
    end
    
    function result = evaluate(obj,xtrain,ytrain,xtest,ytest,kernel_arg,threshold_arg)
      % ----------------------------------------------------------------------------------
      % Avalia o algoritmo knfst
      % ----------------------------------------------------------------------------------      
      % Converte as classes de treinamento para uma indexção sequencial.
      % Ex: se as classes treinadas forem 2,3,5,6 a nova indexação
      % será 1,2,3 e 4. Classes não treinadas recebem rótulo -1.
      ctrain = unique(ytrain);
      ytrain_k = zeros(size(ytrain));
      for i=1:numel(ctrain)
        ytrain_k(ytrain==ctrain(i),1) = i;
      end
      ytest_k = -ones(numel(ytest),1);
      for i=1:numel(ctrain)
        ytest_k(ytest==ctrain(i),1) = i;
      end
      
      % Matriz de Kernel Treinamento x Treinamento
      KTr = obj.kernelMatrix(xtrain,xtrain,kernel_arg);
      model = learn_multiClassNovelty_knfst(KTr,ytrain_k);
      
      % Matriz de Kernel Treinamento x Teste
      KTe = obj.kernelMatrix(xtrain,xtest,kernel_arg);
      [scores,predictions_k] = test_multiClassNovelty_knfst(model, KTe);
      
      dist_target_points = zeros(size(model.target_points,1), size(model.target_points,1));
      for i=1:size(model.target_points,1)
        for j=1:size(model.target_points,1)
          dist_target_points(i,j) = sqrt(sum((model.target_points(i,:) - model.target_points(j,:)).^2));
        end
      end
      
      min_dist = min(dist_target_points(dist_target_points>0));
      dist_threshold = threshold_arg * min_dist;
      
      predictions_k(scores >= dist_threshold) = -1;
      
      % Converte para a indexação real das classes.
      predictions = -ones(size(predictions_k));
      for i=1:numel(ctrain)
        predictions(predictions_k==i) = ctrain(i);
      end
      
      % Report outliers
      outlier_gt = -ones(size(ytest));
      outlier_gt(ytest>0) = 1;
      
      outlier_predictions = -ones(size(predictions));
      outlier_predictions(predictions>0) = 1;
      
      report_outliers = ClassificationReport(outlier_gt,outlier_predictions);
      
      % General report
      report = ClassificationReport(ytest,predictions);
      
      result.kernel =  kernel_arg;
      result.threshold =  threshold_arg;
      result.predictions = predictions;
      result.outlier_predictions = outlier_predictions;
      result.TPR = report_outliers.TPR(2);
      result.TNR = report_outliers.TNR(2);
      result.FPR = report_outliers.FPR(2);
      result.FNR = report_outliers.FNR(2);
      result.F1 = report_outliers.F1(2);
      result.MCC = report_outliers.MCC(2);
      result.ACC = report_outliers.ACC(2);
      result.AFR = report_outliers.AFR(2);
      result.general_conf_matrix = report.CM;
      result.outlier_conf_matrix = report_outliers.CM;
      
      fprintf('\n\nkernel: %f \nthreshold: %f \nTPR: %f \nTNR: %f \nFPR: %f \nFNR: %f \nF1: %f \nMCC: %f \nACC: %f\nAFR: %f\n',...
        kernel_arg,threshold_arg,report_outliers.TPR(2),report_outliers.TNR(2),report_outliers.FPR(2),report_outliers.FNR(2),report_outliers.F1(2),report_outliers.MCC(2),report_outliers.ACC(2),report_outliers.AFR(2));
    end
    
    function result = evaluateAux(obj,KTr,ytrain,KVa,ytest,kernel_arg,threshold_arg)
      % ----------------------------------------------------------------------------------
      % Avalia o algoritmo knfst passsando como entrada as matrizes de
      % kernels
      % ----------------------------------------------------------------------------------      
      % Converte as classes de treinamento para uma indexção sequencial.
      % Ex: se as classes treinadas forem 2,3,5,6 a nova indexação
      % será 1,2,3 e 4. Classes não treinadas recebem rótulo -1.
      ctrain = unique(ytrain);
      ytrain_k = zeros(size(ytrain));
      for i=1:numel(ctrain)
        ytrain_k(ytrain==ctrain(i),1) = i;
      end
      ytest_k = -ones(numel(ytest),1);
      for i=1:numel(ctrain)
        ytest_k(ytest==ctrain(i),1) = i;
      end
      
      model = learn_multiClassNovelty_knfst(KTr,ytrain_k);
      [scores,predictions_k] = test_multiClassNovelty_knfst(model,KVa);
      %pcolor(reshape(scores,[200,200]);
      
      dist_target_points = zeros(size(model.target_points,1), size(model.target_points,1));
      for i=1:size(model.target_points,1)
        for j=1:size(model.target_points,1)
          dist_target_points(i,j) = sqrt(sum((model.target_points(i,:) - model.target_points(j,:)).^2));
        end
      end
      
      min_dist = min(dist_target_points(dist_target_points>0));
      dist_threshold = threshold_arg * min_dist;
      
      predictions_k(scores >= dist_threshold) = -1;
      
      % Converte para a indexação real das classes.
      predictions = -ones(size(predictions_k));
      for i=1:numel(ctrain)
        predictions(predictions_k==i) = ctrain(i);
      end
      
      % Report outliers
      outlier_gt = -ones(size(ytest));
      outlier_gt(ytest>0) = 1;
      
      outlier_predictions = -ones(size(predictions));
      outlier_predictions(predictions>0) = 1;
      
      report_outliers = ClassificationReport(outlier_gt,outlier_predictions);
      
      % General report
      report = ClassificationReport(ytest,predictions);
      
      result.kernel =  kernel_arg;
      result.threshold =  threshold_arg;
      result.predictions = predictions;
      result.outlier_predictions = outlier_predictions;
      result.TPR = report_outliers.TPR(2);
      result.TNR = report_outliers.TNR(2);
      result.FPR = report_outliers.FPR(2);
      result.FNR = report_outliers.FNR(2);
      result.F1 = report_outliers.F1(2);
      result.MCC = report_outliers.MCC(2);
      result.ACC = report_outliers.ACC(2);
      result.AFR = report_outliers.AFR(2);
      result.general_conf_matrix = report.CM;
      result.outlier_conf_matrix = report_outliers.CM;
      
      fprintf('\n\nkernel: %f \nthreshold: %f \nTPR: %f \nTNR: %f \nFPR: %f \nFNR: %f \nF1: %f \nMCC: %f \nACC: %f\nAFR: %f\n',...
        kernel_arg,threshold_arg,report.TPR(2),report_outliers.TNR(2),report_outliers.FPR(2),report_outliers.FNR(2),report_outliers.F1(2),report_outliers.MCC(2),report_outliers.ACC(2),report_outliers.AFR(2));
    end
    
    function predictions = predict(obj,xtrain,ytrain,xtest,kernel_arg,threshold_arg)
      % ----------------------------------------------------------------------------------
      % Avalia o algoritmo knfst
      % ----------------------------------------------------------------------------------      
      % Converte as classes de treinamento para uma indexção sequencial.
      % Ex: se as classes treinadas forem 2,3,5,6 a nova indexação
      % será 1,2,3 e 4. Classes não treinadas recebem rótulo -1.
      ctrain = unique(ytrain);
      ytrain_k = zeros(size(ytrain));
      for i=1:numel(ctrain)
        ytrain_k(ytrain==ctrain(i),1) = i;
      end
      
      % Matriz de Kernel Treinamento x Treinamento
      KTr = obj.kernelMatrix(xtrain,xtrain,kernel_arg);
      model = learn_multiClassNovelty_knfst(KTr,ytrain_k);
      
      % Matriz de Kernel Treinamento x Teste
      KTe = obj.kernelMatrix(xtrain,xtest,kernel_arg);
      [scores,predictions_k] = test_multiClassNovelty_knfst(model, KTe);
      
      dist_target_points = zeros(size(model.target_points,1), size(model.target_points,1));
      for i=1:size(model.target_points,1)
        for j=1:size(model.target_points,1)
          dist_target_points(i,j) = sqrt(sum((model.target_points(i,:) - model.target_points(j,:)).^2));
        end
      end
      
      min_dist = min(dist_target_points(dist_target_points>0));
      dist_threshold = threshold_arg * min_dist;
      
      predictions_k(scores >= dist_threshold) = -1;
      
      % Converte para a indexação real das classes.
      predictions = -ones(size(predictions_k));
      for i=1:numel(ctrain)
        predictions(predictions_k==i) = ctrain(i);
      end
    end
    
    function K = kernelMatrix(obj,X1,X2,kernel_arg)
      % ----------------------------------------------------------------------------------
      % Matriz de kernel
      % ----------------------------------------------------------------------------------      
      if strcmp(obj.kernel_type,'poly')
        offset = 0.5;
        K = (X1*X2' + offset).^kernel_arg;
      else
        gamma = 1/(2*kernel_arg^2);
        K = zeros(size(X1,1),size(X2,1));
        for i=1:size(X1,1)
          for j=1:size(X2,1)
            K(i,j) =  exp(-gamma*norm(X1(i,:) - X2(j,:),2));
          end
        end
      end
    end
  end
end
