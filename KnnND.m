classdef KnnND < handle
  % --------------------------------------------------------------------------------------
  % KNN Novelty Detection for multi-class classification problems.
  %
  % Version 2.0, July 2022.
  % By Samuel Silva (samuelrs@usp.br).
  % --------------------------------------------------------------------------------------
  properties
    X = [];                  % Data matrix [num_samples x dimension]
    Y = [];                  % Labels [num_samples x 1]
    num_samples = 0;         % Number of samples in dataset
    dimension = 0;           % Data dimension
    num_classes = 0;         % Number of classes
    untrained_classes = 0;   % Number of untrained classes
    knn_arg = 0;             % K parameter described in the published paper
    knn_threshold = 0;       % kappa parameter described in the published paper
    num_thresholds = 0;      % Number of "tau" thresholds
    threshold = [];          % "tau" thresholds (the best needs to be found)
    training_ratio = 0;      % Training sample rate
    split = {};              % Holds a split object that helps the cross-validation process
    samples_per_classe = []; % Samples per class
  end
  
  methods
    function obj = KnnND(X,Y,knn_arg,knn_threshold,num_classes,untrained_classes,training_ratio)
      % ----------------------------------------------------------------------------------
      % Constructor.
      %
      % Args
      %   X: Data matrix [num_samples x dimension].
      %   Y: Labels [num_samples x 1].
      %   knn_arg: K parameter described in the published paper.
      %   knn_threshold: kappa parameter described in the published paper.
      %   num_classes: Number of classes in the dataset.
      %   untrained_classes: Number of untrained classes, this parameter can
      %                      be used to simulate novelty data in the dataset.
      %   training_ratio: Training sample rate.
      % ----------------------------------------------------------------------------------
      obj.X = X;
      obj.Y = Y;
      obj.knn_arg = knn_arg;
      obj.knn_threshold = knn_threshold;
      obj.num_classes = num_classes;
      obj.training_ratio = 0.7;
      if nargin>=6
        obj.untrained_classes = untrained_classes;
        if nargin==7
          obj.training_ratio = training_ratio;
        end
      end
      obj.samples_per_classe = sum(Y==unique(Y)',1);
      [obj.samples_per_classe,id] = sort(obj.samples_per_classe,'descend');
      obj.samples_per_classe = cat(1,id,obj.samples_per_classe);
    end
    
    % S� FALTA CHECAR ESSE M�TODO
    function experiment = runExperiments(obj,num_experiments,plot_metric)
      %-----------------------------------------------------------------------------------
      % Runs novelty detection experiments and hyperparameter search.
      %
      % Input args
      %   num_experiments:
      %   plot_metric: if true plots de accuracy metric.
      %
      % Output args
      %   experiments:
      % ----------------------------------------------------------------------------------
      split_exp = cell(num_experiments,1);
      
      MCC = zeros(num_experiments,obj.num_thresholds);
      AFR = zeros(num_experiments,obj.num_thresholds);
      F1 = zeros(num_experiments,obj.num_thresholds);
      TPR = zeros(num_experiments,obj.num_thresholds);
      TNR = zeros(num_experiments,obj.num_thresholds);
      FPR = zeros(num_experiments,obj.num_thresholds);
      FNR = zeros(num_experiments,obj.num_thresholds);
      
      evaluations = cell(num_experiments,obj.num_thresholds);
      
      classes_id = 1:obj.num_classes;
      random_select_classes = true;
      for i=1:num_experiments
        rng(i);
        % Seleciona classes treinadas e n�o treinadas
        if random_select_classes
          [trained,untrained,is_trained_class] = Split.selectClasses(...
            obj.num_classes,obj.untrained_classes);
        else
          % Classe n�o treinada
          classe_unt = rem(i-1,obj.num_classes)+1;
          
          is_trained_class = true(1,obj.num_classes);
          is_trained_class(classe_unt) = false;
          
          trained =  classes_id(classes_id ~= classe_unt);
          untrained =  classes_id(classes_id == classe_unt);
        end
        
        % Divide os �ndices em treino e teste
        [idx_train,idx_test] = Split.trainTestIdx(...
          obj.X,obj.Y,obj.training_ratio,obj.num_classes,is_trained_class);
        [xtrain,xtest,ytrain,ytest] = Split.dataTrainTest(idx_train,idx_test,obj.X,obj.Y);
        
        % Todas as amostras n�o treinadas s�o definidas
        % como outliers (label -1)
        ytest(logical(sum(ytest==untrained,2))) = -1;
        
        RT = [];
        for j=1:obj.num_thresholds
          fprintf('\nKNN (K=%d kappa=%d) \tTest %d/%d \tThreshold %d/%d\n',...
            obj.knn_arg,obj.knn_threshold,i,num_experiments,j,obj.num_thresholds);
          evaluations{i,j} = obj.evaluate(xtrain,ytrain,xtest,ytest,obj.threshold(j));
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
            plot(obj.threshold(1:j),RT,'-','LineWidth',3);
            xlim([obj.threshold(1),obj.threshold(end)]);
            ylim([0,1]);
            xlabel('Threshold');
            ylabel('Matthews correlation coefficient (MCC)');
            title(['KNN [ test ',num2str(i),'/',num2str(num_experiments),' | threshold ',...
              num2str(j),'/',num2str(obj.num_thresholds),' ]']);
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
      mean_mcc = mean(MCC,1);
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
      experiment.all_metrics = all_metrics;
      
      model.training_ratio = obj.training_ratio;
      model.best_threshold_id = best_threshold_id;
      model.threshold = obj.threshold(best_threshold_id);
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
      
      experiment.mcc_score = mean_mcc(best_threshold_id);
      experiment.f1_score = mean_f1(best_threshold_id);
      experiment.afr_score = mean_afr(best_threshold_id);
      experiment.tpr_score = mean_tpr(best_threshold_id);
      experiment.tnr_score = mean_tnr(best_threshold_id);
      experiment.fpr_score = mean_fpr(best_threshold_id);
      experiment.fnr_score = mean_fnr(best_threshold_id);
      
      fprintf('\nRESULTS\n MCC Score: %.4f\n F1 Score: %.4f\n AFR Score: %.4f\n',...
        experiment.mcc_score,experiment.f1_score,experiment.afr_score);
      
      figure;
      plot(obj.threshold,mean_mcc);
      xlim([obj.threshold(1),obj.threshold(end)]);
      xlabel('threshold');
      ylabel('mcc');
      title('MCC');
      
      figure;
      plot(obj.threshold,mean_afr);
      xlim([obj.threshold(1),obj.threshold(end)]);
      xlabel('threshold');
      ylabel('afr');
      title('AFR');
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
      mcc = zeros(num_validations,obj.num_thresholds);
      for i=1:num_validations
        rng(i);
        % Cria um objeto split. Particiona a base em dois conjuntos
        % de classes treinadas e n�o treinadas. Separa uma
        % parte para treinamento e outra para teste
        obj.split{i} = SplitData(obj.X,obj.Y,obj.training_ratio,obj.untrained_classes);
        % Separa uma parte do treinamento para valida��o
        [id_train,id_val] = obj.split{i}.idTrainVal();
        [xtrain,ytrain,xval,yval] = obj.split{i}.dataTrainVal(id_train,id_val);
        RT = [];
        for j=1:obj.num_thresholds
          fprintf('\nKNN (K=%d kappa=%d) \tVal %d/%d \tThreshold %d/%d\n',obj.knn_arg,obj.knn_threshold,i,num_validations,j,obj.num_thresholds);
          result = obj.evaluate(xtrain,ytrain,xval,yval,obj.threshold(j));
          mcc(i,j) = result.MCC;
          if plot_error
            RT = cat(1,RT,mcc(i,j));
            figure(1);
            clf('reset');
            plot(obj.threshold(1:j),RT,'-','LineWidth',3);
            xlim([obj.threshold(1),obj.threshold(end)]);
            ylim([0,1]);
            xlabel('Threshold');
            ylabel('Matthews correlation coefficient (MCC)');
            title(['KNN [ valida��o ',num2str(i),'/',num2str(num_validations),' | threshold ',num2str(j),'/',num2str(obj.num_thresholds),' ]']);
            drawnow;
            pause(0.01);
          end
        end
        model.split{i} = obj.split{i};
      end
      close all;
      mean_mcc = mean(mcc,1);
      [max_mean_mcc,ID] = max(mean_mcc);
      
      model.training_ratio = obj.training_ratio;
      model.threshold = obj.threshold(ID);
      model.untrained_classes = obj.untrained_classes;
      model.knn_arg = obj.knn_arg;
      model.knn_threshold = obj.knn_threshold;
      model.mean_mcc = max_mean_mcc;
    end
    
    function [results,evaluations] = evaluateModel(obj,model,num_tests)
      % ----------------------------------------------------------------------------------
      % Evaluates the KNN prediction with multi-class novelty detection on a trained model.
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
        fprintf('\nKNN (K=%d kappa=%d) \tTest: %d/%d\n',obj.knn_arg,obj.knn_threshold,i,num_tests);
        id_test = obj.split{i}.idTest();
        [xtest,ytest] = obj.split{i}.dataTest(id_test);
        [xtrain,ytrain] = obj.split{i}.dataTrain(obj.split{i}.id_train_val_t);
        evaluations{i} = obj.evaluate(xtrain,ytrain,xtest,ytest,model.threshold);
      end
      results = struct2table(cell2mat(evaluations));
    end
    
    function [results,evaluations] = evaluateTests(obj,xtrain,ytrain,xtest,ytest,model)
      % ----------------------------------------------------------------------------------
      % Evaluates the KNN prediction with multi-class novelty detection on test sets.
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
        fprintf('\nKNN (K=%d kappa=%d) \tTest: %d/%d\n',obj.knn_arg,obj.knn_threshold,i,num_tests);
        evaluations{i} = obj.evaluate(xtrain,ytrain,xtest(:,:,i),ytest,model.threshold);
      end
      results = struct2table(cell2mat(evaluations));
    end
    
    function result = evaluate(obj,xtrain,ytrain,xtest,ytest,threshold)
      % ----------------------------------------------------------------------------------
      % Evaluates the KNN prediction with multi-class novelty detection.
      %
      % Input args
      %   xtrain: training data [num_train x dimensions].
      %   ytrain: training labels [num_train x 1].
      %   xtest: test data [num_test x dimensions].
      %   ytest: test labels [num_test x 1].
      %   threshold: kappa threshold parameter.
      %
      % Output args
      %   result: metrics report for multi-class prediction and novelty detection.
      % ----------------------------------------------------------------------------------
      predictions = obj.predict(xtrain,ytrain,xtest,threshold);
      
      % Report outliers
      outlier_gt = -ones(size(ytest));
      outlier_gt(ytest>0) = 1;
      
      outlier_predictions = -ones(size(predictions));
      outlier_predictions(predictions>0) = 1;
      
      report_outliers = MetricsReport(outlier_gt,outlier_predictions);
      
      % General report
      report = MetricsReportReport(ytest,predictions);
      
      result.threshold =  threshold;
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
      
      fprintf('\nthreshold: %f \nTPR: %f \nTNR: %f \nFPR: %f \nFNR: %f \nF1: %f \nMCC: %f \nACC: %f\nAFR: %f\n',threshold,report_outliers.TPR(2),report_outliers.TNR(2),report_outliers.FPR(2),report_outliers.FNR(2),report_outliers.F1(2),report_outliers.MCC(2),report_outliers.ACC(2),report_outliers.AFR(2));
    end
    
    function predictions = predict(obj,xtrain,ytrain,xtest,threshold)
      % ----------------------------------------------------------------------------------
      % KNN prediction with multi-class novelty detection.
      %
      % Input args
      %   xtrain: training data [num_train x dimensions].
      %   ytrain: training labels [num_train x 1].
      %   xtest: test data [num_test x dimensions].
      %   threshold: kappa threshold parameter.
      %
      % Output args:
      %   predictions: prediction with multi-class novelty detection.
      % ----------------------------------------------------------------------------------
      [epsilons,means,medians,stds,iqrs,maxs] =  obj.computeEpsilons(xtrain,ytrain);
      %epsilons =  medians + threshold * iqrs;
      %epsilons =  means + threshold * stds;
      epsilons =  threshold * epsilons;
      
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
          if n_nearest_neighbor >= obj.knn_threshold
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
      % Computes the epsilon parameters of each class in the training set. This parameter
      % measures the spread of each class. This means that more compact classes have smaller
      % epsilons while more spread classes tend to have larger epsilons.
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
