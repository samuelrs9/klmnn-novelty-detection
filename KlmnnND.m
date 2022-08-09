classdef KlmnnND < handle
  % --------------------------------------------------------------------------------------
  % KLMNN Novelty Detection for multi-class classification problems.
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
    num_untrained_classes = 0;        % number of untrained classes
    knn_arg = 0;                  % K parameter described in the published paper
    kappa_threshold = 0;          % kappa parameter described in the published paper
    num_decision_thresholds = 0;  % number of decision thresholds
    decision_thresholds = [];     % decision thresholds list (the best needs to be found)
    decision_threshold = [];      % decision thresholds
    training_ratio = 0;           % training sample rate
    split = {};                   % holds a split object that helps the cross-validation process
    samples_per_classe = [];      % samples per class
    kernel_type = [];        % kernel function type
    num_kernels = 0;         % number of kernel values
    kernel = [];             % kernel list for kpca algorithm (the best must be found)
  end
  
  methods
    function obj = KlmnnND(X,Y,knn_arg,kappa_threshold,decision_threshold)
      % ----------------------------------------------------------------------------------
      % Constructor.
      %
      % Input args
      %   X: samples [num_samples x dimension].
      %   Y: sample labels [num_samples x 1].
      %   knn_arg: K parameter described in the published paper.
      %   kappa_threshold: kappa parameter described in the published paper.
      %   decision_threshold: decision threshold hyperparameter.
      % ----------------------------------------------------------------------------------]
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
    end
    
    function experiment = runExperiments(obj,num_experiments,random_select_classes,plot_metric)
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
      
      MCC = zeros(obj.num_kernels,num_untrained_classes,num_experiments);
      AFR = zeros(obj.num_kernels,num_untrained_classes,num_experiments);
      F1 = zeros(obj.num_kernels,num_untrained_classes,num_experiments);
      TPR = zeros(obj.num_kernels,num_untrained_classes,num_experiments);
      TNR = zeros(obj.num_kernels,num_untrained_classes,num_experiments);
      FPR = zeros(obj.num_kernels,num_untrained_classes,num_experiments);
      FNR = zeros(obj.num_kernels,num_untrained_classes,num_experiments);
      
      evaluations = cell(obj.num_kernels,num_untrained_classes,num_experiments);
      
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
        
        % Divide os �ndices em treino e teste
        [idx_train,idx_test] = SimpleSplit.trainTestIdx(obj.X,obj.Y,training_ratio,obj.num_classes,is_trained_class);
        [xtrain,xtest,ytrain,ytest] = SimpleSplit.dataTrainTest(idx_train,idx_test,obj.X,obj.Y);
        
        % Todas as amostras de classes n�o treinadas s�o definidas
        % como outliers (label -1)
        ytest(logical(sum(ytest==untrained,2))) = -1;
        
        RK = [];
        for j=1:numel(hyperparameters.kernels)
          kernel_arg = obj.kernel(j);
          
          % Pr�-processamento para o KPCA
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
          
          % Pr�-processamento para o LMNN
          % treino
          mean_trainp = mean(xtrainp);
          xtrainp = xtrainp - mean_trainp;
          max_trainp = max(xtrainp(:));
          xtrainp = xtrainp/max_trainp;
          % teste
          xtestp = xtestp - mean_trainp;
          xtestp = xtestp/max_trainp;
          
          % LMNN
          lmnn = LmnnNovDetection(xtrainp,ytrain);
          T = lmnn.computeTransform(xtrainp,ytrain);
          xtrainpg = lmnn.transform(xtrainp,T);
          xtestpg = lmnn.transform(xtestp,T);
          
          % KNN
          knn = KnnNovDetection(xtrainpg,ytrain);
          RT = [];
          obj.knn_arg = hyperparameters.knn_arg;
          obj.kappa_threshold = hyperparameters.kappa_threshold;          
          for k=1:hyperparameters.num_decision_thresholds
            fprintf('\nKLMNN (K=%d kappa=%d) \tTest: %d/%d \tKernel (%d/%d) \tDecision threshold (%d/%d)\n',...
              obj.knn_arg,obj.kappa_threshold,i,num_experiments,j,obj.num_kernels,k,hyperparameters.num_decision_thresholds);
            evaluations{j,k,i} = knn.evaluate(xtrainpg,ytrain,xtestpg,ytest,hyperparameters.decision_thresholds(k));
            evaluations{j,k,i}.kernel = kernel_arg;
            evaluations{j,k,i}.kpca_model = kpca;
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
              plot(hyperparameters.decision_thresholds(1:k),RT,'-r','LineWidth',3);
              xlim([hyperparameters.decision_thresholds(1),hyperparameters.decision_thresholds(end)]);
              ylim([0,1]);
              xlabel('Threshold');
              ylabel('Matthews correlation coefficient (MCC)');
              title(['KLMNN [ test ',num2str(i),'/',num2str(num_experiments),' | kernel ',num2str(j),'/',num2str(obj.num_kernels),' | decision_threshold ',num2str(k),'/',num2str(num_untrained_classes),' ]']);
              drawnow;
              pause(0.0001);
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
            title(['KLMNN [ test ',num2str(i),'/',num2str(num_experiments),' | kernel ',num2str(j),'/',num2str(obj.num_kernels),' ]']);
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
      % M�trica MCC
      mean_mcc = mean(MCC,3);
      max_mean_mcc = max(max(mean_mcc));
      [best_kernel_id,best_threshold_id] = find(mean_mcc == max_mean_mcc);
      best_kernel_id = best_kernel_id(1);
      best_threshold_id = best_threshold_id(1);
      
      % Demais m�tricas
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
      
      model.training_ratio = training_ratio;
      model.best_threshold_id = best_threshold_id;
      model.best_kernel_id = best_kernel_id;
      model.decision_threshold = hyperparameters.decision_thresholds(best_threshold_id);
      model.kernel = obj.kernel(best_kernel_id);
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
      
      figure; pcolor(hyperparameters.decision_thresholds,obj.kernel,mean_mcc); colorbar;
      xlabel('decision_threshold'); ylabel('kernel');  title('MCC');
      
      figure; pcolor(hyperparameters.decision_thresholds,obj.kernel,mean_afr); colorbar;
      xlabel('decision_threshold'); ylabel('kernel'); title('AFR');
    end
        
    function model = validation(obj,n_validations,plot_error)
      %-----------------------------------------------------------------------------------
      % Valida��o do algoritmo klmnn out detection
      %-----------------------------------------------------------------------------------
      obj.split = cell(n_validations,1);
      mcc = zeros(obj.num_kernels,num_untrained_classes,n_validations);
      for i=1:n_validations
        rng(i);
        % Cria um objeto split. Particiona a base em dois conjuntos
        % de classes treinadas e n�o treinadas. Separa uma
        % parte para treinamento e outra para teste
        obj.split{i} = SplitData(obj.X,obj.Y,training_ratio,num_untrained_classes);
        % Separa uma parte do treinamento para valida��o
        [id_train,id_val] = obj.split{i}.idTrainVal();
        [xtrain,ytrain,xval,yval] = obj.split{i}.dataTrainVal(id_train,id_val);
        RK = [];
        for j=1:obj.num_kernels
          kernel_arg = obj.kernel(j);
          
          % Pr�-processamento para o KPCA
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
          
          % Pr�-processamento para o LMNN
          % treino
          mean_trainp = mean(xtrainp);
          xtrainp = xtrainp - mean_trainp;
          max_trainp = max(xtrainp(:));
          xtrainp = xtrainp/max_trainp;
          % teste
          xvalp = xvalp - mean_trainp;
          xvalp = xvalp/max_trainp;
          
          % LMNN
          lmnn = LmnnNovDetection(xtrainp,ytrain,obj.knn_arg,obj.kappa_threshold,obj.num_classes,num_untrained_classes);
          T = lmnn.computeTransform(xtrainp,ytrain);
          xtrainpg = lmnn.transform(xtrainp,T);
          xvalpg = lmnn.transform(xvalp,T);
          
          % KNN
          knn = KnnNovDetection(xtrainpg,ytrain,obj.knn_arg,obj.kappa_threshold,obj.num_classes,num_untrained_classes);
          RT = [];
          for k=1:num_untrained_classes
            fprintf('\nKLMNN (K=%d kappa=%d) \tVal: %d/%d \tKernel (%d/%d) \tDecision threshold (%d/%d)\n',obj.knn_arg,obj.kappa_threshold,i,n_validations,j,obj.num_kernels,k,num_untrained_classes);
            result = knn.evaluate(xtrainpg,ytrain,xvalpg,yval,hyperparameters.decision_thresholds(k));
            result.kernel = kernel_arg;
            mcc(j,k,i) = result.MCC;
            if plot_error
              RT = cat(1,RT,mcc(j,k,i));
              figure(1);
              clf('reset');
              plot(hyperparameters.decision_thresholds(1:k),RT,'-r','LineWidth',3);
              xlim([hyperparameters.decision_thresholds(1),hyperparameters.decision_thresholds(end)]);
              ylim([0,1]);
              xlabel('Threshold');
              ylabel('Matthews correlation coefficient (MCC)');
              title(['KLMNN [ valida��o ',num2str(i),'/',num2str(n_validations),' | kernel ',num2str(j),'/',num2str(obj.num_kernels),' | decision_threshold ',num2str(k),'/',num2str(num_untrained_classes),' ]']);
              drawnow;
              pause(0.01);
            end
          end
          if plot_error
            RK = cat(1,RK,max(RT));
            figure(2);
            clf('reset');
            plot(obj.kernel(1:j),RK,'-','LineWidth',3);
            xlim([obj.kernel(1),obj.kernel(end)]);
            ylim([0,1]);
            xlabel('Kernel');
            ylabel('Matthews correlation coefficient (MCC)');
            title(['KLMNN [ valida��o ',num2str(i),'/',num2str(n_validations),' | kernel ',num2str(j),'/',num2str(obj.num_kernels),' ]']);
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
      
      model.training_ratio = training_ratio;
      model.kernel = obj.kernel(id_k);
      model.decision_threshold = hyperparameters.decision_thresholds(id_t);
      model.num_untrained_classes = num_untrained_classes;
      model.knn_arg = obj.knn_arg;
      model.kappa_threshold = obj.kappa_threshold;
      model.mean_mcc = max_mean_mcc;
    end
    
    function [results,evaluations] = evaluateModel(obj,model,num_tests)
      % ----------------------------------------------------------------------------------
      % This method is used to evaluate the KLMNN prediction with multi-class novelty 
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
        fprintf('\nKLMNN (K=%d kappa=%d) \tTest: %d/%d\n',obj.knn_arg,obj.kappa_threshold,i,num_tests);
        id_test = obj.split{i}.idTest();
        [xtest,ytest] = obj.split{i}.dataTest(id_test);
        [xtrain,ytrain] = obj.split{i}.dataTrain(obj.split{i}.id_train_val_t);
        evaluations{i} = obj.evaluate(xtrain,ytrain,xtest,ytest,model.kernel,model.decision_threshold);
        evaluations{i}.kernel = model.kernel;
      end
      results = struct2table(cell2mat(evaluations));
    end
    
    function [results,evaluations] = evaluateTests(obj,xtrain,ytrain,xtest,ytest,model)
      % ----------------------------------------------------------------------------------
      % This method is used to evaluate the KLMNN prediction with multi-class novelty 
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
        fprintf('\nKLMNN (K=%d kappa=%d) \tTest: %d/%d\n',obj.knn_arg,obj.kappa_threshold,i,num_tests);
        evaluations{i} = obj.evaluate(xtrain,ytrain,xtest(:,:,i),ytest,model.kernel,model.decision_threshold);
        evaluations{i}.kernel = model.kernel;
      end
      results = struct2table(cell2mat(evaluations));
    end
    
    function result = evaluate(obj,xtrain,ytrain,xtest,ytest,kernel_arg,threshold_arg)
      % ----------------------------------------------------------------------------------
      % This method is used to evaluate the KLMNN prediction with multi-class novelty detection.
      %
      % Input args
      %   xtrain: training data [num_train x dimensions].
      %   ytrain: training labels [num_train x 1].
      %   xtest: test data [num_test x dimensions].
      %   ytest: test labels [num_test x 1].
      %   kernel_arg: kernel parameter for kpca algorithm.
      %   threshold_arg: kappa decision_threshold parameter.
      %
      % Output args
      %   result: metrics report for multi-class prediction and novelty detection.
      % ----------------------------------------------------------------------------------      
      % Pr�-processamento para o KPCA
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
      
      % Pr�-processamento para o LMNN
      % treino
      mean_trainp = mean(xtrainp);
      xtrainp = xtrainp - mean_trainp;
      max_trainp = max(xtrainp(:));
      xtrainp = xtrainp/max_trainp;
      % teste
      xtestp = xtestp - mean_trainp;
      xtestp = xtestp/max_trainp;
      
      % LMNN
      lmnn = LmnnNovDetection(xtrainp,ytrain,obj.knn_arg,obj.kappa_threshold,obj.num_classes,num_untrained_classes);
      T = lmnn.computeTransform(xtrainp,ytrain);
      xtrainpg = lmnn.transform(xtrainp,T);
      xtestpg = lmnn.transform(xtestp,T);
      
      % KNN
      knn = KnnNovDetection(xtrainpg,ytrain,obj.knn_arg,obj.kappa_threshold,obj.num_classes,num_untrained_classes);
      result = knn.evaluate(xtrainpg,ytrain,xtestpg,ytest,threshold_arg);
      result.kpca_model = kpca;
    end
    
    function predictions = predict(obj,xtrain,ytrain,xtest,kernel_arg,threshold_arg)
      % ----------------------------------------------------------------------------------
      % This method is used to run KLMNN prediction with multi-class novelty detection.
      %
      % Input args
      %   xtrain: training data [num_train x dimensions].
      %   ytrain: training labels [num_train x 1].
      %   xtest: test data [num_test x dimensions].
      %   kernel_arg: kernel parameter.
      %   threshold_arg: kappa decision_threshold parameter.
      %
      % Output args:
      %   predictions: prediction with multi-class novelty detection.
      % ----------------------------------------------------------------------------------      
      % Pr�-processamento para o KPCA
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
      
      % Pr�-processamento para o LMNN
      % treino
      mean_trainp = mean(xtrainp);
      xtrainp = xtrainp - mean_trainp;
      max_trainp = max(xtrainp(:));
      xtrainp = xtrainp/max_trainp;
      % teste
      xtestp = xtestp - mean_trainp;
      xtestp = xtestp/max_trainp;
      
      % LMNN
      lmnn = LmnnNovDetection(xtrainp,ytrain,obj.knn_arg,obj.kappa_threshold,obj.num_classes,num_untrained_classes);
      T = lmnn.computeTransform(xtrainp,ytrain);
      xtrainpg = lmnn.transform(xtrainp,T);
      xtestpg = lmnn.transform(xtestp,T);
      
      % KNN
      knn = KnnNovDetection(xtrainpg,ytrain,obj.knn_arg,obj.kappa_threshold,obj.num_classes,num_untrained_classes);
      predictions = knn.predict(xtrainpg,ytrain,xtestpg,threshold_arg);
    end
    
    function model = kpcaModel(obj,kernel_arg)
      % ----------------------------------------------------------------------------------
      % This method build a Kernel PCA model.
      % 
      % Input arg
      %   kernel_arg: kernel parameter of kernel pca algorithm.
      % 
      % Output args
      %   model: kernel pca model.
      % ----------------------------------------------------------------------------------
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


