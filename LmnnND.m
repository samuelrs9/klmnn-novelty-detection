classdef LmnnND < KnnND
  % --------------------------------------------------------------------------------------
  % LMNN Novelty Detection for multi-class classification problems.
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
    max_iter = 500;          % maximum number of iterations of the lmnn algorithm
  end
  
  methods
    function obj = LmnnND(X,Y,knn_arg,knn_threshold,num_classes,untrained_classes,training_ratio)
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
      obj.max_iter = 500;
    end
    
    % SÓ FALTA CHECAR ESSE MÉTODO
    function experiment = runExperiments(obj,num_experiments,view_plot_metric)
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
        knn = KnnNovDetection(xtraing,ytrain,obj.knn_arg,obj.knn_threshold,obj.num_classes,obj.untrained_classes);
        
        RT = [];
        for j=1:obj.num_thresholds
          fprintf('\nLMNN (K=%d kappa=%d) \tTest %d/%d \tThreshold %d/%d\n',obj.knn_arg,obj.knn_threshold,i,num_experiments,j,obj.num_thresholds);
          evaluations{i,j} = knn.evaluate(xtraing,ytrain,xtestg,ytest,obj.threshold(j));
          MCC(i,j) = evaluations{i,j}.MCC;
          F1(i,j) = evaluations{i,j}.F1;
          AFR(i,j) = evaluations{i,j}.AFR;
          TPR(i,j) = evaluations{i,j}.TPR;
          TNR(i,j) = evaluations{i,j}.TNR;
          FPR(i,j) = evaluations{i,j}.FPR;
          FNR(i,j) = evaluations{i,j}.FNR;
          if view_plot_metric
            RT = cat(1,RT,MCC(i,j));
            figure(1);
            clf('reset');
            plot(obj.threshold(1:j),RT,'-','LineWidth',3);
            xlim([obj.threshold(1),obj.threshold(end)]);
            ylim([0,1]);
            xlabel('Threshold');
            ylabel('Matthews correlation coefficient (MCC)');
            title(['LMNN [ test ',num2str(i),'/',num2str(num_experiments),' | threshold ',num2str(j),'/',num2str(obj.num_thresholds),' ]']);
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
      
      figure; plot(obj.threshold,mean_mcc);
      xlim([obj.threshold(1),obj.threshold(end)]);
      xlabel('threshold'); ylabel('mcc'); title('MCC');
      
      figure; plot(obj.threshold,mean_afr);
      xlim([obj.threshold(1),obj.threshold(end)]);
      xlabel('threshold'); ylabel('afr'); title('AFR');
    end
    
    % Validação do algoritmo lmnn out detection
    function model = validation(obj,num_validations,view_plot_error)
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
        % de classes treinadas e não treinadas. Separa uma
        % parte para treinamento e outra para teste
        obj.split{i} = SplitData(obj.X,obj.Y,obj.training_ratio,obj.untrained_classes);
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
        knn = KnnNovDetection(xtraing,ytrain,obj.knn_arg,obj.knn_threshold,obj.num_classes,obj.untrained_classes);
        RT = [];
        for j=1:obj.num_thresholds
          fprintf('\nLMNN (K=%d kappa=%d) \tVal %d/%d \tThreshold %d/%d\n',obj.knn_arg,obj.knn_threshold,i,num_validations,j,obj.num_thresholds);
          
          result = knn.evaluate(xtraing,ytrain,xvalg,yval,obj.threshold(j),a);
          
          mcc(i,j) = result.MCC;
          if view_plot_error
            RT = cat(1,RT,mcc(i,j));
            figure(1);
            clf('reset');
            plot(obj.threshold(1:j),RT,'-','LineWidth',3);
            xlim([obj.threshold(1),obj.threshold(end)]);
            ylim([0,1]);
            xlabel('Threshold');
            ylabel('Matthews correlation coefficient (MCC)');
            title(['LMNN [ test ',num2str(i),'/',num2str(num_validations),' | threshold ',num2str(j),'/',num2str(obj.num_thresholds),' ]']);
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
    
    % Avalia o modelo treinado
    function [results,evaluations] = evaluateModel(obj,model,n_tests)
      evaluations = cell(n_tests,1);
      for i=1:n_tests
        rng(i);
        fprintf('\nLMNN (K=%d kappa=%d) \tTest: %d/%d\n',obj.knn_arg,obj.knn_threshold,i,n_tests);
        id_test = obj.split{i}.idTest();
        [xtest,ytest] = obj.split{i}.dataTest(id_test);
        [xtrain,ytrain] = obj.split{i}.dataTrain(obj.split{i}.id_train_val_t);
        evaluations{i} = obj.evaluate(xtrain,ytrain,xtest,ytest,model.threshold);
      end
      results = struct2table(cell2mat(evaluations));
    end
    
    % Avalia o modelo treinado em conjuntos de testes
    function [results,evaluations] = evaluateTests(obj,xtrain,ytrain,xtest,ytest,model)
      n_tests = size(xtest,3);
      evaluations = cell(n_tests,1);
      for i=1:n_tests
        fprintf('\nLMNN (K=%d kappa=%d) \tTest: %d/%d\n',obj.knn_arg,obj.knn_threshold,i,n_tests);
        evaluations{i} = obj.evaluate(xtrain,ytrain,xtest(:,:,i),ytest,model.threshold);
      end
      results = struct2table(cell2mat(evaluations));
    end
    
    % CONTINUAR DAQUI
    function result = evaluate(obj,xtrain,ytrain,xtest,ytest,threshold,transform)
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
      %knn = KnnNovDetection(xtraing,ytrain,obj.knn_arg,obj.knn_threshold,obj.num_classes,obj.untrained_classes);
      result = evaluate@KnnND(obj,xtraing,ytrain,xtestg,ytest,threshold);
    end
    
    % Prediction
    function predictions = predict(obj,xtrain,ytrain,xtest,threshold)
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
      knn = KnnNovDetection(xtraing,ytrain,obj.knn_arg,obj.knn_threshold,obj.num_classes,obj.untrained_classes);
      predictions = knn.predict(xtraing,ytrain,xtestg,threshold);
    end
    
    % Treina o algoritmo lmnn
    function T = computeTransform(obj,X,Y)
      fprintf('\nComputing LMNN...\n');
      [T,~] = lmnnCG(X',Y,3,'maxiter',obj.max_iter);
      %[~,T,~,~] = lmnn(X,Y,obj.knn_arg);
    end
    
    % Aplica a transformação
    function Tx = transform(obj,x,T)
      Tx = T*x';
      Tx = Tx';
    end
  end
end

