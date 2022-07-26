classdef LmnnND < handle
  % --------------------------------------------------------------------------------------
  % LMNN Novelty Detection for multi-class classification problems.
  %
  % Version 2.0, July 2022.
  % By Samuel Silva (samuelrs@usp.br).
  % --------------------------------------------------------------------------------------
  properties
    X = [];                       % data samples [num_samples x dimension]
    Y = [];                       % labels [num_samples x 1]
    num_samples = 0;              % number of samples in dataset
    dimension = 0;                % data dimension
    num_classes = 0;              % number of classes    
    knn_arg = 0;                  % K parameter described in the published paper
    kappa_threshold = 0;          % kappa parameter described in the published paper
    decision_threshold = 0;       % decision thresholds
    samples_per_classe = [];      % samples per class
    max_iter = 500;               % maximum number of iterations of the lmnn algorithm
  end
  
  methods
    function obj = LmnnND(X,Y,knn_arg,kappa_threshold,decision_threshold)
      % ----------------------------------------------------------------------------------
      % Constructor.
      %
      % Input args
      %   X: data samples [num_samples x dimension].
      %   Y: labels [num_samples x 1].
      %   knn_arg: K parameter described in the published paper.
      %   kappa_threshold: kappa parameter described in the published paper.   
      %   decision_threshold: decision threshold hyperparameter.
      % ----------------------------------------------------------------------------------
      obj.X = X;
      obj.Y = Y;      
      if nargin>=3
        obj.knn_arg = knn_arg;
      else
        obj.knn_arg = 5;
      end
      if nargin>=4
        obj.kappa_threshold = kappa_threshold;
      else
        obj.kappa_threshold = 2;
      end
      if nargin==5
        obj.decision_threshold = decision_threshold;
      else
        obj.decision_threshold = 1.2;
      end
      obj.num_classes = numel(unique(Y));
      obj.samples_per_classe = sum(Y==unique(Y)',1);
      [obj.samples_per_classe,id] = sort(obj.samples_per_classe,'descend');
      obj.samples_per_classe = cat(1,id,obj.samples_per_classe);
      obj.max_iter = 500;
    end
    
    function experiments = runExperiments(obj,hyperparameters,num_experiments,...
      num_untrained_classes,training_ratio,random_select_classes,plot_metric)
      %-----------------------------------------------------------------------------------
      % This method runs validation experiments and hyperparameter search.
      %
      % Input args
      %   hyperparameters: a struct containing the hyperparameters, such as 'knn_arg', 
      %     'kappa_threshold' and 'decision_thresholds' candidates.
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
      obj.knn_arg = hyperparameters.knn_arg;
      obj.kappa_threshold = hyperparameters.kappa_threshold;        
      num_decision_thresholds = hyperparameters.num_decision_thresholds;
      decision_thresholds = hyperparameters.decision_thresholds;
      
      split_exp = cell(num_experiments,1);
      
      MCC = zeros(num_experiments,num_decision_thresholds);
      AFR = zeros(num_experiments,num_decision_thresholds);
      F1 = zeros(num_experiments,num_decision_thresholds);
      TPR = zeros(num_experiments,num_decision_thresholds);
      TNR = zeros(num_experiments,num_decision_thresholds);
      FPR = zeros(num_experiments,num_decision_thresholds);
      FNR = zeros(num_experiments,num_decision_thresholds);
      
      evaluations = cell(num_experiments,num_decision_thresholds);      
          
      t0_lmnn = tic;
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
        
        % Split indices into training and testing indices
        [idx_train,idx_test] = SimpleSplit.trainTestIdx(obj.X,obj.Y,training_ratio,is_trained_class);
        [xtrain,xtest,ytrain,ytest] = SimpleSplit.dataTrainTest(idx_train,idx_test,obj.X,obj.Y);
        
        % All untrained samples are defined as outliers (label = -1)
        ytest(logical(sum(ytest==untrained,2))) = -1;
        
        % Preprocessing for LMNN
        % training
        mean_train = mean(xtrain);
        xtrain = xtrain - mean_train;
        max_train = max(xtrain(:));
        xtrain = xtrain/max_train;
        % teste
        xtest = xtest - mean_train;
        xtest = xtest/max_train;
        
        % LMNN
        T = obj.computeTransform(xtrain,ytrain);
        xtraing = obj.transform(xtrain,T);
        xtestg = obj.transform(xtest,T);
                
        % KNN
        knn = KnnND(xtraing,ytrain,obj.knn_arg,obj.kappa_threshold);
        knn.num_classes = obj.num_classes; % Number of classes of the original dataset
        
        RT = [];
        for j=1:num_decision_thresholds
            fprintf('\nLMNN (K=%d kappa=%d) \tTest %d/%d \tDecision threshold %d/%d\n',...
            obj.knn_arg,obj.kappa_threshold,i,num_experiments,j,num_decision_thresholds);          
          evaluations{i,j} = knn.evaluate(xtraing,ytrain,xtestg,ytest,decision_thresholds(j));
          MCC(i,j) = evaluations{i,j}.MCC;
          F1(i,j) = evaluations{i,j}.F1;
          AFR(i,j) = evaluations{i,j}.AFR;
          TPR(i,j) = evaluations{i,j}.TPR;
          TNR(i,j) = evaluations{i,j}.TNR;
          FPR(i,j) = evaluations{i,j}.FPR;
          FNR(i,j) = evaluations{i,j}.FNR;
          if plot_metric
            RT = cat(1,RT,MCC(i,j));
            figure(1);
            clf('reset');
            plot(decision_thresholds(1:j),RT,'-','LineWidth',2);
            xlim([decision_thresholds(1),decision_thresholds(end)]);
            ylim([0,1]);
            xlabel('Threshold');
            ylabel('Matthews correlation coefficient (MCC)');
            title(['LMNN [ test ',num2str(i),'/',num2str(num_experiments),' | decision-threshold ',...
              num2str(j),'/',num2str(num_decision_thresholds),' ]']);
            drawnow;
            pause(0.01);
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
      % M�trica MCC
      mean_mcc = mean(MCC,1,'omitnan');
      [~,best_threshold_id] = max(mean_mcc);
      
      % Demais m�tricas
      mean_f1 = mean(F1,1);
      mean_afr = mean(AFR,1);
      mean_tpr = mean(TPR,1);
      mean_tnr = mean(TNR,1);
      mean_fpr = mean(FPR,1);
      mean_fnr = mean(FNR,1);
      
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
      model.num_decision_thresholds = num_decision_thresholds;
      
      model.knn_arg = obj.knn_arg;
      model.kappa_threshold = obj.kappa_threshold;
      model.decision_threshold = decision_thresholds(best_threshold_id);
      
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
      
      experiments.mcc_score = mean_mcc(best_threshold_id);
      experiments.f1_score = mean_f1(best_threshold_id);
      experiments.afr_score = mean_afr(best_threshold_id);
      experiments.tpr_score = mean_tpr(best_threshold_id);
      experiments.tnr_score = mean_tnr(best_threshold_id);
      experiments.fpr_score = mean_fpr(best_threshold_id);
      experiments.fnr_score = mean_fnr(best_threshold_id);
      
      experiments.total_time = toc(t0_lmnn);    
      
      fprintf('\nRESULTS\n MCC Score: %.4f\n F1 Score: %.4f\n AFR Score: %.4f\n',...
        experiments.mcc_score,experiments.f1_score,experiments.afr_score);
      
      figure; 
      plot(decision_thresholds,mean_mcc,'LineWidth',2);
      xlim([decision_thresholds(1),decision_thresholds(end)]);
      xlabel('decision-threshold'); 
      ylabel('mcc'); 
      title('MCC');
      
      figure; 
      plot(decision_thresholds,mean_f1,'LineWidth',2);
      xlim([decision_thresholds(1),decision_thresholds(end)]);
      xlabel('decision-threshold'); 
      ylabel('f1-score'); 
      title('F1-SCORE');
    end

    function [results,evaluations] = evaluateTests(obj,xtrain,ytrain,xtest,ytest,model)
      % ----------------------------------------------------------------------------------
      % This method is used to evaluate the LMNN prediction with multi-class novelty 
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
        fprintf('\nLMNN (K=%d kappa=%d) \tTest: %d/%d\n',obj.knn_arg,obj.kappa_threshold,i,num_tests);
        evaluations{i} = obj.evaluate(xtrain,ytrain,xtest(:,:,i),ytest,model.decision_threshold);
      end
      results = struct2table(cell2mat(evaluations));
    end
    
    function result = evaluate(obj,xtrain,ytrain,xtest,ytest,decision_threshold)
      % ----------------------------------------------------------------------------------
      % This method is used to evaluate the LMNN prediction with multi-class novelty detection.
      %
      % Input args
      %   xtrain: training data [num_train x dimensions].
      %   ytrain: training labels [num_train x 1].
      %   xtest: test data [num_test x dimensions].
      %   ytest: test labels [num_test x 1].
      %   decision_threshold: decision threshold hyperparameter.
      %
      % Output args
      %   result: metrics report for multi-class prediction and novelty detection.
      % ----------------------------------------------------------------------------------
      % Pr�-processamento para o LMNN
      % treino
      mean_train = mean(xtrain);
      xtrain = xtrain - mean_train;
      max_train = max(xtrain(:));
      xtrain = xtrain/max_train;
      % teste
      xtest = xtest - mean_train;
      xtest = xtest/max_train;
      
      % LMNN
      T = obj.computeTransform(xtrain,ytrain);
      xtraing = obj.transform(xtrain,T);
      xtestg = obj.transform(xtest,T);
      
      % KNN
      knn = KnnND(xtraing,ytrain,obj.knn_arg,obj.kappa_threshold);
      result = knn.evaluate(xtraing,ytrain,xtestg,ytest,decision_threshold);
      %result = evaluate@KnnND(obj,xtraing,ytrain,xtestg,ytest,decision_threshold);
    end
    
    function predictions = predict(obj,xtrain,ytrain,xtest,decision_threshold)
      % ----------------------------------------------------------------------------------
      % This method is used to run LMNN prediction with multi-class novelty detection.
      %
      % Input args
      %   xtrain: training data [num_train x dimensions].
      %   ytrain: training labels [num_train x 1].
      %   xtest: test data [num_test x dimensions].
      %   decision_threshold: decisino threshold hyparameter.
      %
      % Output args:
      %   predictions: prediction with multi-class novelty detection.
      % ----------------------------------------------------------------------------------      
      % Pr�-processamento
      % treino
      mean_train = mean(xtrain);
      xtrain = xtrain - mean_train;
      max_train = max(xtrain(:));
      xtrain = xtrain/max_train;
      % teste
      xtest = xtest - mean_train;
      xtest = xtest/max_train;
      
      % LMNN
      T = obj.computeTransform(xtrain,ytrain);
      xtraing = obj.transform(xtrain,T);
      xtestg = obj.transform(xtest,T);
      
      % Visualization.map(xtrain,xtrainp,ytrain)
      
      % KNN
      knn = KnnND(xtraing,ytrain,obj.knn_arg,obj.kappa_threshold);
      predictions = knn.predict(xtraing,ytrain,xtestg,decision_threshold);
    end
    
    function T = computeTransform(obj,xtrain,ytrain)
      % ----------------------------------------------------------------------------------
      % This method is used to run metric learning using LMNN algorithm.
      %
      % Input args
      %   xtrain: training data [num_train x dimensions].
      %   ytrain: training labels [num_train x 1].
      %
      % Output args:
      %   T: linear transformation that improves the KNN classifier.
      % ----------------------------------------------------------------------------------                  
      fprintf('\nComputing LMNN...\n');
      [T,~] = lmnnCG(xtrain',ytrain,3,'maxiter',obj.max_iter);
    end
        
    function data_t = transform(obj,data,T)
      % ----------------------------------------------------------------------------------                  
      % This method applies the linear transformation found by the LMNN algorithm.
      %
      % Input args
      %   data: data to be transformed [num_data x dimensions].
      %   T: linear transformation found by LMNN algorithm.
      %
      % Output args:
      % data_t: data transformed by linear transformation T.
      % ----------------------------------------------------------------------------------                  
      data_t = T*data';
      data_t = data_t';
    end
  end
end

