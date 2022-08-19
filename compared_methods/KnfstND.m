classdef KnfstND < handle
  % --------------------------------------------------------------------------------------
  % KNFST Novelty Detection for multi-class classification problems.
  %
  % Version 2.0, July 2022.
  % By Samuel Silva (samuelrs@usp.br).
  % --------------------------------------------------------------------------------------  
  properties
    X = [];                       % samples [num_samples x dimension]
    Y = [];                       % sample labels [num_samples x 1]
    num_samples = 0;              % number of samples in dataset
    dimension = 0;                % data dimension
    num_classes = 0;              % number of classes    
    decision_threshold = [];      % decision threshold
    samples_per_classe = [];      % samples per class
    kernel_type = [];             % kernel function type
    kernel = 0;                  % kernel list for svm algorithm (the best must be found)    
  end
  
  methods
    function obj = KnfstND(X,Y,decision_threshold,kernel)
      % ----------------------------------------------------------------------------------
      % Constructor.
      %
      % Input args
      %   X: samples [num_samples x dimension].
      %   Y: sample labels [num_samples x 1].
      %   decision_threshold: decision threshold hyperparameter.
      %   kernel: kernel hyperparameter for knfst algorithm.
      % ----------------------------------------------------------------------------------]
      obj.X = X;
      obj.Y = Y;      
      if nargin>=3
        obj.decision_threshold = decision_threshold;
      else
          obj.decision_threshold = 1.2;
      end      
      if nargin==4
        obj.kernel = kernel;
      else
          obj.kernel = 1.0;
      end            
      obj.num_classes = numel(unique(Y));
      obj.samples_per_classe = sum(Y==unique(Y)',1);
      [obj.samples_per_classe,id] = sort(obj.samples_per_classe,'descend');
      obj.samples_per_classe = cat(1,id,obj.samples_per_classe);      
    end

    function experiments = runExperiments(obj,hyperparameters,num_experiments,...
      num_untrained_classes,training_ratio,random_select_classes,plot_metric)
      %-----------------------------------------------------------------------------------
      % This method runs validation experiments and hyperparameter search.
      %
      % Input args
      %   hyperparameters: a struct containing the hyperparameters, such as 
      %     'decision_thresholds' candidates and 'kernels' candidates.     
      %   num_experiments: number of validation experiments.      
      %   num_untrained_classes: number of untrained classes, this parameter can
      %     be used to simulate novelty data in the dataset.
      %   training_ratio: training sample rate.      
      %   random_select_classes: enable/disable random selection of untrained classes (a
      %     boolean value).
      %   plot_metric: enable/disable the accuracy metrics plot (a boolean value).
      %
      % Output args
      %   experiments: experiments report.
      % ----------------------------------------------------------------------------------         
      obj.kernel_type = hyperparameters.kernel_type;
      num_decision_thresholds = hyperparameters.num_decision_thresholds;
      decision_thresholds = hyperparameters.decision_thresholds;
      num_kernels = hyperparameters.num_kernels;
      kernels = hyperparameters.kernels;
      
      split_exp = cell(num_experiments,1);
      
      MCC = zeros(num_kernels,num_decision_thresholds,num_experiments);
      AFR = zeros(num_kernels,num_decision_thresholds,num_experiments);
      F1 = zeros(num_kernels,num_decision_thresholds,num_experiments);
      TPR = zeros(num_kernels,num_decision_thresholds,num_experiments);
      TNR = zeros(num_kernels,num_decision_thresholds,num_experiments);
      FPR = zeros(num_kernels,num_decision_thresholds,num_experiments);
      FNR = zeros(num_kernels,num_decision_thresholds,num_experiments);
      
      evaluations = cell(num_kernels,num_decision_thresholds,num_experiments);
      
      t0_knfst = tic;
      for i=1:num_experiments
        rng(i);
        if random_select_classes
          % Randomly selects trained and untrained classes
          [trained,untrained,is_trained_class] = SimpleSplit.selectClasses(...
            obj.num_classes,num_untrained_classes);
        else
          % Use the class -1 as novelty.
          % First reset the label -1 to "max(classes)+1",
          % this is done for compatibility with the code present in the Split class
          classes = unique(obj.Y);
          classes(classes==-1) = max(classes)+1;
          classes = sort(classes);
          obj.Y(obj.Y==-1) = classes(end);
          
          is_trained_class = true(1,obj.num_classes);
          is_trained_class(end) = false;
          
          trained =  classes(1:end-1);
          untrained =  classes(end);
        end

        % In each experiment selects only one untrained class
        [idx_train,idx_test] = SimpleSplit.trainTestIdx(obj.X,obj.Y,training_ratio,is_trained_class);
        [xtrain,xtest,ytrain,ytest] = SimpleSplit.dataTrainTest(idx_train,idx_test,obj.X,obj.Y);
        
        % All untrained samples are defined as outliers (label -1)
        ytest(logical(sum(ytest==untrained,2))) = -1;
        
        RK = [];
        for j=1:num_kernels
          kernel_arg = kernels(j);
          
          % Matriz de Kernel Treinamento x Treinamento
          KTr = obj.kernelMatrix(xtrain,xtrain,kernel_arg);
          
          % Matriz de Kernel Treinamento x Teste
          KTe = obj.kernelMatrix(xtrain,xtest,kernel_arg);
          
          RT = [];
          for k=1:num_decision_thresholds
            fprintf('\nKNFST \tTest: %d/%d \tKernel (%d/%d) \tDecision threshold (%d/%d)\n',i,num_experiments,j,num_kernels,k,num_decision_thresholds);
            decision_threshold = decision_thresholds(k);
            evaluations{j,k,i} = obj.evaluateAux(KTr,ytrain,KTe,ytest,kernel_arg,decision_threshold);
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
              plot(decision_thresholds(1:k),RT,'-r','LineWidth',3);
              xlim([decision_thresholds(1),decision_thresholds(end)]);
              ylim([0,1]);
              xlabel('Threshold');
              ylabel('Matthews correlation coefficient (MCC)');
              title(['KNFST [ test ',num2str(i),'/',num2str(num_experiments),' | kernel ',num2str(j),'/',num2str(num_kernels),' | decision_threshold ',num2str(k),'/',num2str(num_decision_thresholds),' ]']);
              drawnow;
              pause(0.01);
            end
          end
          if plot_metric
            RK = cat(1,RK,max(RT));
            figure(2);
            clf('reset');
            plot(kernels(1:j),RK,'-','LineWidth',3);
            xlim([kernels(1),kernels(end)]);
            ylim([0,1]);
            xlabel('Kernel');
            ylabel('Matthews correlation coefficient (MCC)');
            title(['KNFST [ test ',num2str(i),'/',num2str(num_experiments),' | kernel ',num2str(j),'/',num2str(num_kernels),' ]']);
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
      mean_mcc = mean(MCC,3,'omitnan');
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
      experiments.all_metrics = all_metrics;
      
      model.training_ratio = training_ratio;
      model.best_threshold_id = best_threshold_id;
      model.best_kernel_id = best_kernel_id;
      model.num_untrained_classes = num_untrained_classes;      
      model.decision_threshold = decision_thresholds(best_threshold_id);
      model.kernel = kernels(best_kernel_id);
      model.kernel_type = obj.kernel_type;    
      
      experiments.hyperparameters = hyperparameters;      
      experiments.num_experiments = num_experiments;      

      experiments.model = model;
      experiments.split = cell2mat(split_exp);
      experiments.evaluations = evaluations;
      experiments.mean_mcc = mean_mcc;
      experiments.mean_f1 = mean_f1;
      experiments.mean_afr = mean_afr;
      experiments.mean_tpr = mean_tpr;
      experiments.mean_tnr = mean_tnr;
      experiments.mean_fpr = mean_fpr;
      experiments.mean_fnr = mean_fnr;
      
      experiments.mcc_score = mean_mcc(best_kernel_id,best_threshold_id);
      experiments.f1_score = mean_f1(best_kernel_id,best_threshold_id);
      experiments.afr_score = mean_afr(best_kernel_id,best_threshold_id);
      experiments.tpr_score = mean_tpr(best_kernel_id,best_threshold_id);
      experiments.tnr_score = mean_tnr(best_kernel_id,best_threshold_id);
      experiments.fpr_score = mean_fpr(best_kernel_id,best_threshold_id);
      experiments.fnr_score = mean_fnr(best_kernel_id,best_threshold_id);
      
      experiments.total_time = toc(t0_knfst);
      
      fprintf('\nRESULTS\n MCC Score: %.4f\n F1 Score: %.4f\n AFR Score: %.4f\n',...
        experiments.mcc_score,experiments.f1_score,experiments.afr_score);
      
      figure;  
      pcolor(decision_thresholds,kernels,mean_mcc); 
      colorbar;
      xlabel('decision_threshold'); 
      ylabel('kernel'); 
      title('MCC');
      
      figure; 
      pcolor(decision_thresholds,kernels,mean_f1); 
      colorbar;
      xlabel('decision_threshold'); 
      ylabel('kernel'); 
      title('F1-SCORE');
    end
    
    function [results,evaluations] = evaluateModel(obj,model,num_tests)
      % ----------------------------------------------------------------------------------
      % This method is used to evaluate the KNFST prediction with multi-class novelty 
      % detection using a trained model.
      %
      % Input args
      %   model: trained model.
      %   num_tests: number of tests to be performed.
      %
      % Output args
      %   [results,evaluations]: metrics report for multi-class prediction and novelty detection.
      % -----------------------------------------------------------------------------------
      evaluations = cell(num_tests,1);
      for i=1:num_tests
        rng(i);
        fprintf('\nKNFST Test: %d/%d\n',i,num_tests);
        id_test = obj.split{i}.idTest();
        [xtest,ytest] = obj.split{i}.dataTest(id_test);
        [xtrain,ytrain] = obj.split{i}.dataTrain(obj.split{i}.id_train_val_t);
        evaluations{i} = obj.evaluate(xtrain,ytrain,xtest,ytest,model.kernel,model.decision_threshold);
      end
      results = struct2table(cell2mat(evaluations));
    end

    function [results,evaluations] = evaluateTests(obj,xtrain,ytrain,xtest,ytest,model)
      % ----------------------------------------------------------------------------------
      % This method is used to evaluate the KNFST prediction with multi-class novelty 
      % detection on test sets.
      %
      % Input args
      %   xtrain: training data [num_train x dimensions].
      %   ytrain: training labels [num_train x 1].
      %   xtest: test data [num_test x dimensions].
      %   ytest: test labels [num_test x 1].
      %   model: trained model.
      %
      % Output args
      %   [results,evaluations]: metrics report for multi-class prediction and novelty detection.
      % ----------------------------------------------------------------------------------
      num_tests = size(xtest,3);
      evaluations = cell(num_tests,1);
      for i=1:num_tests
        fprintf('\nKNFST \tTest: %d/%d\n',i,num_tests);
        evaluations{i} = obj.evaluate(xtrain,ytrain,xtest(:,:,i),ytest,model.kernel,model.decision_threshold);
      end
      results = struct2table(cell2mat(evaluations));
    end
    
    function result = evaluate(obj,xtrain,ytrain,xtest,ytest,kernel_arg,decision_threshold)
      % ----------------------------------------------------------------------------------
      % This method is used to evaluate the KNFST prediction with multi-class novelty detection.
      %
      % Input args
      %   xtrain: training data [num_train x dimensions].
      %   ytrain: training labels [num_train x 1].
      %   xtest: test data [num_test x dimensions].
      %   ytest: test labels [num_test x 1].
      %   kernel_arg: kernel parameter.
      %   decision_threshold: decision threshold hyperparameter.
      %
      % Output args
      %   result: metrics report for multi-class prediction and novelty detection.
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
      dist_threshold = decision_threshold * min_dist;
      
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
      
      report_outliers = MetricsReport(outlier_gt,outlier_predictions);
      
      % General report
      report = MetricsReport(ytest,predictions);
      
      result.kernel =  kernel_arg;
      result.decision_threshold =  decision_threshold;
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
        kernel_arg,decision_threshold,report_outliers.TPR(2),report_outliers.TNR(2),report_outliers.FPR(2),report_outliers.FNR(2),report_outliers.F1(2),report_outliers.MCC(2),report_outliers.ACC(2),report_outliers.AFR(2));
    end
    
    function result = evaluateAux(obj,KTr,ytrain,KTe,ytest,kernel_arg,decision_threshold)
      % ----------------------------------------------------------------------------------
      % This method is used to evaluate the KNFST prediction with multi-class novelty 
      % detection in validation experiments.
      %
      % Input args
      %   KTr: training kernel matrix [num_train x num_train].
      %   ytrain: training labels [num_train x 1].
      %   KTe: test kernel matrix [num_train x num_test].
      %   ytest: test labels [num_test x 1].
      %   kernel_arg: kernel parameter.
      %   decision_threshold: decision decision_threshold parameter.
      %
      % Output args
      %   result: metrics report for multi-class prediction and novelty detection.
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
      [scores,predictions_k] = test_multiClassNovelty_knfst(model,KTe);
      %pcolor(reshape(scores,[200,200]);
      
      dist_target_points = zeros(size(model.target_points,1), size(model.target_points,1));
      for i=1:size(model.target_points,1)
        for j=1:size(model.target_points,1)
          dist_target_points(i,j) = sqrt(sum((model.target_points(i,:) - model.target_points(j,:)).^2));
        end
      end
      
      min_dist = min(dist_target_points(dist_target_points>0));
      dist_threshold = decision_threshold * min_dist;
      
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
      
      report_outliers = MetricsReport(outlier_gt,outlier_predictions);
      
      % General report
      report = MetricsReport(ytest,predictions);
      
      result.kernel =  kernel_arg;
      result.decision_threshold =  decision_threshold;
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
        kernel_arg,decision_threshold,report.TPR(2),report_outliers.TNR(2),report_outliers.FPR(2),report_outliers.FNR(2),report_outliers.F1(2),report_outliers.MCC(2),report_outliers.ACC(2),report_outliers.AFR(2));
    end
    
    function predictions = predict(obj,xtrain,ytrain,xtest,kernel_arg,decision_threshold)
      % ----------------------------------------------------------------------------------
      % This method is used to run KNFST prediction with multi-class novelty detection.
      %
      % Input args
      %   xtrain: training data [num_train x dimensions].
      %   ytrain: training labels [num_train x 1].
      %   xtest: test data [num_test x dimensions].
      %   kernel_arg: kernel parameter.
      %   decision_threshold: decision threshold parameter.
      %
      % Output args:
      %   predictions: prediction with multi-class novelty detection.
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
      dist_threshold = decision_threshold * min_dist;
      
      predictions_k(scores >= dist_threshold) = -1;
      
      % Converte para a indexação real das classes.
      predictions = -ones(size(predictions_k));
      for i=1:numel(ctrain)
        predictions(predictions_k==i) = ctrain(i);
      end
    end
    
    function K = kernelMatrix(obj,X1,X2,kernel_arg)
      % ----------------------------------------------------------------------------------
      % This method is used to calculate the kernel matrix with respect to two sets of 
      % samples X1 and X2.
      %
      % Input args
      %   X1: samples 1.
      %   X2: samples 2.
      %   kernel_arg: kernel parameter.
      %
      % Output args:
      %   K: kernel matrix.
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
