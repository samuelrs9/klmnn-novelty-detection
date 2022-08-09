classdef LmnnND < handle
  % --------------------------------------------------------------------------------------
  % LMNN Novelty Detection for multi-class classification problems.
  %
  % Version 2.0, July 2022.
  % By Samuel Silva (samuelrs@usp.br).
  % --------------------------------------------------------------------------------------
  properties
    X = [];                  % data samples [num_samples x dimension]
    Y = [];                       % labels [num_samples x 1]
    num_samples = 0;              % number of samples in dataset
    dimension = 0;                % data dimension
    num_classes = 0;              % number of classes    
    knn_arg = 0;                  % K parameter described in the published paper
    kappa_threshold = 0;          % kappa parameter described in the published paper
    decision_threshold = [];      % decision thresholds
    training_ratio = 0;           % training sample rate
    split = {};                   % holds a split object that helps the cross-validation process
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
      if nargin==2
        obj.knn_arg = 5;
        obj.kappa_threshold = 2;
        obj.decision_threshold = 1.2;
      elseif nargin==5
        obj.knn_arg = knn_arg;
        obj.kappa_threshold = kappa_threshold;
        obj.decision_threshold = decision_threshold;
      else 
        error(['Number of input arguments is wrong! You must pass all 3 arguments' ...'
          '(knn_arg, kappa_threshold and decision_threshold) or none of them.']);
      end
      obj.num_classes = numel(unique(Y));
      obj.samples_per_classe = sum(Y==unique(Y)',1);
      [obj.samples_per_classe,id] = sort(obj.samples_per_classe,'descend');
      obj.samples_per_classe = cat(1,id,obj.samples_per_classe);
      obj.max_iter = 500;
    end
    
    function experiment = runExperiments(obj,num_experiments,plot_metric)
      %-----------------------------------------------------------------------------------
      % This method runs validation experiments and hyperparameter search.
      %
      % Input args
      %   num_experiments: number of validation experiments.
      %   random_select_classes: enable/disable random selection of untrained classes (a
      %     boolean value).
      %   plot_metric: enable/disable the accuracy metrics plot (a boolean value).
      %
      % Output args
      %   experiments: experiments report.
      % ----------------------------------------------------------------------------------
      split_exp = cell(num_experiments,1);
      
      MCC = zeros(num_experiments,num_untrained_classes);
      AFR = zeros(num_experiments,num_untrained_classes);
      F1 = zeros(num_experiments,num_untrained_classes);
      TPR = zeros(num_experiments,num_untrained_classes);
      TNR = zeros(num_experiments,num_untrained_classes);
      FPR = zeros(num_experiments,num_untrained_classes);
      FNR = zeros(num_experiments,num_untrained_classes);
      
      evaluations = cell(num_experiments,num_untrained_classes);
      
      for i=1:num_experiments
        rng(i);
        if random_select_classes
          % Randomly selects trained and untrained classes
          [trained,untrained,is_trained_class] = SimpleSplit.selectClasses(...
            obj.num_classes,num_untrained_classes);
        else
          % In each experiment selects only one untrained class
          classe_unt = rem(i-1,obj.num_classes)+1;
          
          is_trained_class = true(1,obj.num_classes);
          is_trained_class(classe_unt) = false;
          
          trained =  classes_id(classes_id ~= classe_unt);
          untrained =  classes_id(classes_id == classe_unt);
        end
        
        % Split indices into training and testing indices
        [idx_train,idx_test] = SimpleSplit.trainTestIdx(obj.X,obj.Y,training_ratio,obj.num_classes,is_trained_class);
        [xtrain,xtest,ytrain,ytest] = SimpleSplit.dataTrainTest(idx_train,idx_test,obj.X,obj.Y);
        
        % All untrained samples are defined
        % as outliers (label = -1)
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
        knn = KnnNovDetection(xtraing,ytrain,obj.knn_arg,obj.kappa_threshold,obj.num_classes,num_untrained_classes);
        
        RT = [];
        obj.knn_arg = hyperparameters.knn_arg;
        obj.kappa_threshold = hyperparameters.kappa_threshold;        
        for j=1:hyperparameters.num_decision_thresholds
          fprintf('\nLMNN (K=%d kappa=%d) \tTest %d/%d \tDecision threshold %d/%d\n',obj.knn_arg,obj.kappa_threshold,i,num_experiments,j,num_untrained_classes);
          evaluations{i,j} = knn.evaluate(xtraing,ytrain,xtestg,ytest,hyperparameters.decision_thresholds(j));
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
            plot(hyperparameters.decision_thresholds(1:j),RT,'-','LineWidth',3);
            xlim([hyperparameters.decision_thresholds(1),hyperparameters.decision_thresholds(end)]);
            ylim([0,1]);
            xlabel('Threshold');
            ylabel('Matthews correlation coefficient (MCC)');
            title(['LMNN [ test ',num2str(i),'/',num2str(num_experiments),' | decision_threshold ',num2str(j),'/',num2str(num_untrained_classes),' ]']);
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
      % Métrica MCC
      mean_mcc = mean(MCC,1);
      [~,best_threshold_id] = max(mean_mcc);
      
      % Demais métricas
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
      experiment.all_metrics = all_metrics;
      
      model.training_ratio = training_ratio;
      model.best_threshold_id = best_threshold_id;
      model.decision_threshold = hyperparameters.decision_thresholds(best_threshold_id);
      model.num_untrained_classes = num_untrained_classes;
      model.knn_arg = obj.knn_arg;
      model.kappa_threshold = obj.kappa_threshold;
      
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
      
      experiment.mcc_score = mean_mcc(best_threshold_id);
      experiment.f1_score = mean_f1(best_threshold_id);
      experiment.afr_score = mean_afr(best_threshold_id);
      experiment.tpr_score = mean_tpr(best_threshold_id);
      experiment.tnr_score = mean_tnr(best_threshold_id);
      experiment.fpr_score = mean_fpr(best_threshold_id);
      experiment.fnr_score = mean_fnr(best_threshold_id);
      
      fprintf('\nRESULTS\n MCC Score: %.4f\n F1 Score: %.4f\n AFR Score: %.4f\n',...
        experiment.mcc_score,experiment.f1_score,experiment.afr_score);
      
      figure; plot(hyperparameters.decision_thresholds,mean_mcc);
      xlim([hyperparameters.decision_thresholds(1),hyperparameters.decision_thresholds(end)]);
      xlabel('decision_threshold'); ylabel('mcc'); title('MCC');
      
      figure; plot(hyperparameters.decision_thresholds,mean_afr);
      xlim([hyperparameters.decision_thresholds(1),hyperparameters.decision_thresholds(end)]);
      xlabel('decision_threshold'); ylabel('afr'); title('AFR');
    end
    
    function model = validation(obj,num_validations,plot_error)
      %-----------------------------------------------------------------------------------
      % Runs a cross-validation algorithm.
      %
      % Input args
      %   num_validations:
      %   plot_error: if true plots de accuracy metric.
      %
      % Output args
      %   model:
      % ----------------------------------------------------------------------------------
      obj.split = cell(num_validations,1);
      mcc = zeros(num_validations,num_untrained_classes);
      for i=1:num_validations
        rng(i);
        % Cria um objeto split. Particiona a base em dois conjuntos
        % de classes treinadas e não treinadas. Separa uma
        % parte para treinamento e outra para teste
        obj.split{i} = SplitData(obj.X,obj.Y,training_ratio,num_untrained_classes);
        % Separa uma parte do treinamento para validação
        [id_train,id_val] = obj.split{i}.idTrainVal();
        [xtrain,ytrain,xval,yval] = obj.split{i}.dataTrainVal(id_train,id_val);
        
        % Pré-processamento para o LMNN
        % treino
        mean_train = mean(xtrain);
        xtrain = xtrain - mean_train;
        max_train = max(xtrain(:));
        xtrain = xtrain/max_train;
        % validação
        xval = xval - mean_train;
        xval = xval/max_train;
        
        % LMNN
        T = obj.computeTransform(xtrain,ytrain);
        xtraing = obj.transform(xtrain,T);
        xvalg = obj.transform(xval,T);
        
        % KNN
        knn = KnnNovDetection(xtraing,ytrain,obj.knn_arg,obj.kappa_threshold,obj.num_classes,num_untrained_classes);
        RT = [];
        for j=1:num_untrained_classes
          fprintf('\nLMNN (K=%d kappa=%d) \tVal %d/%d \tDecision threshold %d/%d\n',obj.knn_arg,obj.kappa_threshold,i,num_validations,j,num_untrained_classes);
          
          result = knn.evaluate(xtraing,ytrain,xvalg,yval,hyperparameters.decision_thresholds(j),a);
          
          mcc(i,j) = result.MCC;
          if plot_error
            RT = cat(1,RT,mcc(i,j));
            figure(1);
            clf('reset');
            plot(hyperparameters.decision_thresholds(1:j),RT,'-','LineWidth',3);
            xlim([hyperparameters.decision_thresholds(1),hyperparameters.decision_thresholds(end)]);
            ylim([0,1]);
            xlabel('Threshold');
            ylabel('Matthews correlation coefficient (MCC)');
            title(['LMNN [ test ',num2str(i),'/',num2str(num_validations),' | decision_threshold ',num2str(j),'/',num2str(num_untrained_classes),' ]']);
            drawnow;
            pause(0.01);
          end
        end
        model.split{i} = obj.split{i};
      end
      close all;
      mean_mcc = mean(mcc,1);
      [max_mean_mcc,ID] = max(mean_mcc);
      
      model.training_ratio = training_ratio;
      model.decision_threshold = hyperparameters.decision_thresholds(ID);
      model.num_untrained_classes = num_untrained_classes;
      model.knn_arg = obj.knn_arg;
      model.kappa_threshold = obj.kappa_threshold;
      model.mean_mcc = max_mean_mcc;
    end
    
    function [results,evaluations] = evaluateModel(obj,model,num_tests)
      % ----------------------------------------------------------------------------------
      % This method is used to evaluate the LMNN prediction with multi-class novelty 
      % detection on a trained model.
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
        fprintf('\nLMNN (K=%d kappa=%d) \tTest: %d/%d\n',obj.knn_arg,obj.kappa_threshold,i,num_tests);
        id_test = obj.split{i}.idTest();
        [xtest,ytest] = obj.split{i}.dataTest(id_test);
        [xtrain,ytrain] = obj.split{i}.dataTrain(obj.split{i}.id_train_val_t);
        evaluations{i} = obj.evaluate(xtrain,ytrain,xtest,ytest,model.decision_threshold);
      end
      results = struct2table(cell2mat(evaluations));
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
      % Pré-processamento para o LMNN
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
      knn = KnnNovDetection(xtraing,ytrain,obj.knn_arg,...
        obj.kappa_threshold,obj.num_classes,num_untrained_classes);
      result = knn.evaluate(obj,xtraing,ytrain,xtestg,ytest,decision_threshold);
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
      % Pré-processamento
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
      knn = KnnNovDetection(xtraing,ytrain,obj.knn_arg,obj.kappa_threshold,obj.num_classes,num_untrained_classes);
      predictions = knn.predict(xtraing,ytrain,xtestg,threshold);
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

