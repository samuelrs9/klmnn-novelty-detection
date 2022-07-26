classdef KnnND < handle
  % --------------------------------------------------------------------------------------
  % KNN Novelty Detection for multi-class classification problems.
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
  end
  
  methods
    function obj = KnnND(X,Y,knn_arg,kappa_threshold,decision_threshold)
      % ----------------------------------------------------------------------------------
      % Constructor.
      %
      % Input args
      %   X: samples [num_samples x dimension].
      %   Y: sample labels [num_samples x 1].
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
      if nargin>=5
        obj.decision_threshold = decision_threshold;
      else
        obj.decision_threshold = 1.2;
      end
      obj.num_classes = numel(unique(Y));
      obj.samples_per_classe = sum(Y==unique(Y)',1);
      [obj.samples_per_classe,id] = sort(obj.samples_per_classe,'descend');
      obj.samples_per_classe = cat(1,id,obj.samples_per_classe);
    end
    
    function experiments = runExperiments(obj,hyperparameters,num_experiments,...
      num_untrained_classes,training_ratio,random_select_classes,plot_metric)
      %-----------------------------------------------------------------------------------
      % This method runs validation experiments and hyperparameter searches.
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
      
      t0_knn = tic;      
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

        % All untrained samples are defined as outliers (label -1)
        ytest(logical(sum(ytest==untrained,2))) = -1;            
        
        RT = [];
        for j=1:num_decision_thresholds
          fprintf('\nKNN (K=%d kappa=%d) \tTest %d/%d \tDecision threshold %d/%d\n',...
            obj.knn_arg,obj.kappa_threshold,i,num_experiments,j,num_decision_thresholds);          
          evaluations{i,j} = obj.evaluate(xtrain,ytrain,xtest,ytest,decision_thresholds(j));
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
            title(['KNN [ test ',num2str(i),'/',num2str(num_experiments),...
              ' | decision threshold ',num2str(j),'/',num2str(num_decision_thresholds),' ]']);
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
      % Finds the decision threshold index that gives the best average matthews correlation 
      % coefficient on all validation experiments
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
      model.num_untrained_classes = num_untrained_classes;
            
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
      
      experiments.total_time = toc(t0_knn);
      
      fprintf('\nRESULTS\n MCC Score: %.4f\n F1 Score: %.4f\n AFR Score: %.4f\n',...
        experiments.mcc_score,experiments.f1_score,experiments.afr_score);
      
      figure;
      plot(decision_thresholds,mean_mcc,'LineWidth',2);
      xlim([decision_thresholds(1),decision_thresholds(end)]);
      xlabel('decision-thresholds');
      ylabel('mcc');
      title('MCC');
      
      figure;
      plot(decision_thresholds,mean_f1,'LineWidth',2);
      xlim([decision_thresholds(1),decision_thresholds(end)]);
      xlabel('decision-thresholds');
      ylabel('f1-score');
      title('F1-SCORE');
    end
    
    function [results,evaluations] = evaluateTests(obj,xtrain,ytrain,xtest,ytest,model)
      % ----------------------------------------------------------------------------------
      % This method is used to evaluate the KNN prediction with multi-class novelty 
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
        fprintf('\nKNN (K=%d kappa=%d) \tTest: %d/%d\n',model.knn_arg,model.kappa_threshold,i,num_tests);
        evaluations{i} = obj.evaluate(xtrain,ytrain,xtest(:,:,i),ytest,model.decision_threshold);
      end
      results = struct2table(cell2mat(evaluations));
    end
    
    function result = evaluate(obj,xtrain,ytrain,xtest,ytest,decision_threshold)
      % ----------------------------------------------------------------------------------
      % This method is used to evaluate the KNN prediction with multi-class novelty detection.
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
      predictions = obj.predict(xtrain,ytrain,xtest,decision_threshold);
      
      % Report outliers
      outlier_gt = -ones(size(ytest));
      outlier_gt(ytest>0) = 1;
      
      outlier_predictions = -ones(size(predictions));
      outlier_predictions(predictions>0) = 1;
      
      report_outliers = MetricsReport(outlier_gt,outlier_predictions);
      
      % General report
      report = MetricsReport(ytest,predictions);
      
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
      
      fprintf('\nthreshold: %f \nTPR: %f \nTNR: %f \nFPR: %f \nFNR: %f \nF1: %f \nMCC: %f \nACC: %f\nAFR: %f\n',...
        decision_threshold,report_outliers.TPR(2),report_outliers.TNR(2),report_outliers.FPR(2),report_outliers.FNR(2),...
        report_outliers.F1(2),report_outliers.MCC(2),report_outliers.ACC(2),report_outliers.AFR(2));
    end
    
    function predictions = predict(obj,xtrain,ytrain,xtest,decision_threshold)
      % ----------------------------------------------------------------------------------
      % This method is used to run KNN prediction with multi-class novelty detection.
      %
      % Input args
      %   xtrain: training data [num_train x dimensions].
      %   ytrain: training labels [num_train x 1].
      %   xtest: test data [num_test x dimensions].
      %   decision_threshold: decision threshold hyperparameter.
      %
      % Output args:
      %   predictions: prediction with multi-class novelty detection.
      % ----------------------------------------------------------------------------------
      [epsilons,means,medians,stds,iqrs,maxs] =  obj.computeEpsilons(xtrain,ytrain);
      %epsilons =  medians + decision_threshold * iqrs;
      %epsilons =  means + decision_threshold * stds;
      epsilons =  decision_threshold * epsilons;
      
      NTe = size(xtest,1);
      
      predictions = zeros(NTe,1);
      
      full_kdtree = KDTreeSearcher(xtrain);
      classes_kdtree = cell(obj.num_classes,1);
      
      % Divide as amostras de treino por classes
      xtrainpart = cell(obj.num_classes,1);
      counterclass = zeros(obj.num_classes,1);
      for c=1:obj.num_classes
        xtrainpart{c}.X = xtrain(ytrain==c,:);
        xtrainpart{c}.n = size(xtrainpart{c}.X,1);
        counterclass(c) = xtrainpart{c}.n;
      end
      
      % Cria k-trees por classe
      for c=1:obj.num_classes
        if xtrainpart{c}.n==0
          continue;
        end
        classes_kdtree{c} = KDTreeSearcher(xtrainpart{c}.X);
      end
      
      % Avalia��o no teste com epsilons
      for i=1:NTe
        % Testa se o exemplo i � outlier
        isoutlier = true;
        for c=1:obj.num_classes
          if xtrainpart{c}.n == 0
            continue;
          end
          [~,d] = knnsearch(classes_kdtree{c},xtest(i,:),'k',obj.knn_arg);
          % N�mero de vizinhos suficientemente pr�ximos
          n_nearest_neighbor = sum(d<epsilons(c));
          if n_nearest_neighbor >= obj.kappa_threshold
            isoutlier = false;
            break;
          end
        end
        
        if(isoutlier)
          predictions(i) = -1;
        else % Inlier
          
          % Classifica o exemplo
          tryK = obj.knn_arg;
          while true
            [n,d] = knnsearch(full_kdtree,xtest(i,:),'k',tryK); % busca os K mais pr�ximos de cada ponto xtest_i
            
            % Classes dos K vizinhos mais pr�ximos
            yvec = ytrain(n);
            
            % Seleciona vizinhos mais pr�ximos v�lidos
            valid_nearest_neighbor = false(numel(yvec),1);
            for k=1:numel(d)
              if d(k) < epsilons(yvec(k))
                valid_nearest_neighbor(k) = true;
              end
            end
            n_valid_nearest_neighbor = sum(valid_nearest_neighbor);
            
            % Se o exemplo de teste n�o foi considerado
            % outlier por�m ficou com uma quantidade de vizinhos
            % mais pr�ximos inferior a K tente buscar 2*K
            % vizinhos mais pr�ximos e repita o processo...
            if n_valid_nearest_neighbor < obj.knn_arg
              % se o vizinho mais distante dista mais que os
              % epsilons de todas as classes, ent�o pare de
              % dobrar o tryK
              if d(end) > max(epsilons) || tryK >= size(xtrain,1)
                yvec = yvec(valid_nearest_neighbor);
                break;
              else
                tryK = 2*tryK;
              end
            else
              yvec = yvec(valid_nearest_neighbor);
              break;
            end
          end
          predictions(i) = mode(yvec);
        end
      end
    end
    
    function [epsilons,means,medians,stds,iqrs,maxs] = computeEpsilons(obj,xtrain,ytrain)
      % ----------------------------------------------------------------------------------
      % This method is used to compute the epsilon parameters of each class in the training 
      % set. This parameter measures the spread of each class. This means that more compact
      % classes have smaller epsilons while more spread classes tend to have larger epsilons.
      %
      % Input args
      %   xtrain: training data [num_train x dimensions].
      %   ytrain: training labels [num_train x 1].
      %
      % Output args
      %   epsilons: epsilon parameters for each class.
      %   means: average distance to k nearest neighbor for each class.
      %   medians: median distance to k nearest neighbor for each class.
      %   stds: standard deviation of the distance to the k-nearest neighbor for each class.
      %   iqrs: interquartile range of distance to the k-nearest neighbor for each class.
      %   maxs: maximum distance to k nearest neighbor for each class.
      % ----------------------------------------------------------------------------------
      NTr=size(xtrain,1);
      
      means = zeros(obj.num_classes,1);
      medians = zeros(obj.num_classes,1);
      iqrs = zeros(obj.num_classes,1);
      stds = zeros(obj.num_classes,1);
      maxs = zeros(obj.num_classes,1);
      epsilons = zeros(obj.num_classes,1);
      
      full_kdtree = KDTreeSearcher(xtrain);
      
      classes_kdtree = cell(obj.num_classes,1);
      xtrainpart = cell(obj.num_classes,1);
      dim = size(xtrain,2);
      
      % Separa o conjunto de treino por classe
      for c=1:obj.num_classes
        xtrainpart{c}.n = 0;
      end
      for i=1:NTr
        xtrainpart{ytrain(i)}.n = xtrainpart{ytrain(i)}.n + 1;
      end
      for c=1:obj.num_classes
        xtrainpart{c}.X = zeros(xtrainpart{c}.n,dim);
      end
      counterclass = zeros(obj.num_classes,1);
      for i=1:NTr
        label = ytrain(i);
        counterclass(label) = counterclass(label) + 1;
        xtrainpart{label}.X(counterclass(label),:) = xtrain(i,:);
      end
      
      % Cria uma kdtree por classe
      for c=1:obj.num_classes
        if xtrainpart{c}.n==0
          continue;
        end
        classes_kdtree{c} = KDTreeSearcher(xtrainpart{c}.X);
      end
      
      % Calcula os epsilons
      kdistsperclass = zeros(obj.num_classes,NTr);
      for c=1:obj.num_classes
        if xtrainpart{c}.n == 0
          continue;
        end
        count = 0;
        for i=1:xtrainpart{c}.n
          [n,d] = knnsearch(classes_kdtree{c},xtrainpart{c}.X(i,:),'k',obj.knn_arg+1);
          count = count + 1;
          kdistsperclass(c,count) = d(end);
        end
        dists = kdistsperclass(c,1:count);
        epsilons(c) = (mean(dists) + std(dists));
        means(c) = mean(dists);
        medians(c) = median(dists);
        iqrs(c) = iqr(dists);
        stds(c) = std(dists);
        maxs(c) = max(dists);
      end
    end
  end
end
