classdef SvmND < handle
  % --------------------------------------------------------------------------------------
  % SVM Novelty Detection for multi-class classification problems.
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
    decision_threshold = 0;       % decision threshold
    samples_per_classe = [];      % samples per class
    kernel_type = [];             % kernel function type
    kernel = [];                  % kernel list for svm algorithm (the best must be found)    
  end
  
  methods
    function obj = SvmND(X,Y,decision_threshold,kernel)
      % ----------------------------------------------------------------------------------
      % Constructor.
      %
      % Input args
      %   X: samples [num_samples x dimension].
      %   Y: sample labels [num_samples x 1].
      %   decision_threshold: decision threshold hyperparameter.
      %   kernel: kernel hyperparameter for svm algorithm.
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

    function experiment = runOneSVMExperiments(obj,num_experiments,plot_metric)
      %-----------------------------------------------------------------------------------
      % This method runs validation experiments and hyperparameter search for OneSVM.
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
      
      MCC = zeros(num_experiments,obj.num_kernels);
      AFR = zeros(num_experiments,obj.num_kernels);
      F1 = zeros(num_experiments,obj.num_kernels);
      TPR = zeros(num_experiments,obj.num_kernels);
      TNR = zeros(num_experiments,obj.num_kernels);
      FPR = zeros(num_experiments,obj.num_kernels);
      FNR = zeros(num_experiments,obj.num_kernels);
      
      evaluations = cell(num_experiments,num_untrained_classes);
      
      for i=1:num_experiments
        rng(i);
        % Seleciona classes treinadas e não treinadas
        [trained,untrained,is_trained_class] = SimpleSplit.selectClasses(obj.num_classes,obj.untrained_classes);
        
        % Divide os índices em treino e teste
        [idx_train,idx_test] = SimpleSplit.trainTestIdx(obj.X,obj.Y,training_ratio,obj.num_classes,is_trained_class);
        [xtrain,xtest,ytrain,ytest] = SimpleSplit.dataTrainTest(idx_train,idx_test,obj.X,obj.Y);
        
        % Todas as amostras não treinadas são definidas
        % como outliers (label -1)
        ytest(logical(sum(ytest==untrained,2))) = -1;
        
        RK = [];
        for j=1:obj.num_kernels
          fprintf('\nOne SVM \tTest %d/%d \tKernel %d/%d\n',i,num_experiments,j,obj.num_kernels);
          evaluations{i,j} = obj.evaluateOneSVM(xtrain,ytrain,xtest,ytest,obj.kernel(j));
          evaluations{i,j}.kernel = obj.kernel(j);
          MCC(i,j) = evaluations{i,j}.MCC;
          F1(i,j) = evaluations{i,j}.F1;
          AFR(i,j) = evaluations{i,j}.AFR;
          TPR(i,j) = evaluations{i,j}.TPR;
          TNR(i,j) = evaluations{i,j}.TNR;
          FPR(i,j) = evaluations{i,j}.FPR;
          FNR(i,j) = evaluations{i,j}.FNR;
          if plot_metric
            RK = cat(1,RK,MCC(i,j));
            figure(1);
            clf('reset');
            plot(obj.kernel(1:j),RK,'-','LineWidth',3);
            xlim([obj.kernel(1),obj.kernel(end)]);
            ylim([0,1]);
            xlabel('Kernel');
            ylabel('Matthews correlation coefficient (MCC)');
            title(['One SVM [ test ',num2str(i),'/',num2str(num_experiments),' | kernel ',num2str(j),'/',num2str(obj.num_kernels),' ]']);
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
      [~,best_kernel_id] = max(mean_mcc);
      
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
      model.best_kernel_id = best_kernel_id;
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
      
      experiment.mcc_score = mean_mcc(best_kernel_id);
      experiment.f1_score = mean_f1(best_kernel_id);
      experiment.afr_score = mean_afr(best_kernel_id);
      experiment.tpr_score = mean_tpr(best_kernel_id);
      experiment.tnr_score = mean_tnr(best_kernel_id);
      experiment.fpr_score = mean_fpr(best_kernel_id);
      experiment.fnr_score = mean_fnr(best_kernel_id);
      
      fprintf('\nRESULTS\n MCC Score: %.4f\n F1 Score: %.4f\n AFR Score: %.4f\n',...
        experiment.mcc_score,experiment.f1_score,experiment.afr_score);
      
      figure; plot(obj.kernel,mean_mcc);
      xlabel('kernel'); ylabel('mcc'); title('MCC');
      
      figure; plot(obj.kernel,mean_afr);
      xlabel('kernel'); ylabel('afr'); title('AFR');
    end

    function experiment = runMultiSVMExperiments(obj,num_experiments,plot_metric)
      %-----------------------------------------------------------------------------------
      % This method runs validation experiments and hyperparameter search for MultiSVM.
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
        % Seleciona classes treinadas e não treinadas
        [trained,untrained,is_trained_class] = SimpleSplit.selectClasses(obj.num_classes,obj.untrained_classes);
        
        % Divide os índices em treino e teste
        [idx_train,idx_test] = SimpleSplit.trainTestIdx(obj.X,obj.Y,training_ratio,obj.num_classes,is_trained_class);
        [xtrain,xtest,ytrain,ytest] = SimpleSplit.dataTrainTest(idx_train,idx_test,obj.X,obj.Y);
        
        % Todas as amostras não treinadas são definidas
        % como outliers (label -1)
        ytest(logical(sum(ytest==untrained,2))) = -1;
        
        RK = [];
        for j=1:obj.num_kernels
          % Treina classificadores SVM's binários com a abordagem One vs All
          ctrain = unique(ytrain);
          model_svm = cell(numel(ctrain),1);
          for c=1:numel(ctrain)
            ytrain_binary = -ones(numel(ytrain),1);
            ytrain_binary(ytrain==ctrain(c)) = 1;
            if strcmp(obj.kernel_type,'poly')
              svm_model = fitcsvm(xtrain,ytrain_binary,'KernelFunction','polynomial', 'PolynomialOrder', obj.kernel(j));
            else
              svm_model = fitcsvm(xtrain,ytrain_binary,'KernelFunction','rbf', 'KernelScale', 1/(2*obj.kernel(j)^2) );
            end
            model_svm{c} = fitSVMPosterior(svm_model);
          end
          
          RT = [];
          for k=1:num_untrained_classes
            fprintf('\nMulti SVM \tTest: %d/%d \tKernel (%d/%d) \tDecision threshold (%d/%d)\n',i,num_experiments,j,obj.num_kernels,k,num_untrained_classes);
            evaluations{j,k,i} = obj.evaluateMultiSVMAux(model_svm,ctrain,xtest,ytest,obj.kernel(j),hyperparameters.decision_thresholds(k));
            evaluations{j,k,i}.kernel = obj.kernel(j);
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
              title(['Multi SVM [ test ',num2str(i),'/',num2str(num_experiments),' | kernel ',num2str(j),'/',num2str(obj.num_kernels),' | decision_threshold ',num2str(k),'/',num2str(num_untrained_classes),' ]']);
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
            title(['Multi SVM [ test ',num2str(i),'/',num2str(num_experiments),' | kernel ',num2str(j),'/',num2str(obj.num_kernels),' ]']);
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
      
      model.training_ratio = training_ratio;
      model.best_threshold_id = best_threshold_id;
      model.best_kernel_id = best_kernel_id;
      model.decision_threshold = hyperparameters.decision_thresholds(best_threshold_id);
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
      
      figure; pcolor(hyperparameters.decision_thresholds,obj.kernel,mean_mcc); colorbar;
      xlabel('decision_threshold'); ylabel('kernel');  title('MCC');
      
      figure; pcolor(hyperparameters.decision_thresholds,obj.kernel,mean_afr); colorbar;
      ('decision_threshold'); ylabel('kernel'); title('AFR');
    end
    
    function model = validationOneClass(obj,n_validations,plot_error)
      % ----------------------------------------------------------------------------------
      % Validação do algoritmo one class svm
      % ----------------------------------------------------------------------------------      
      obj.split = cell(n_validations,1);
      mcc = zeros(n_validations,obj.num_kernels);
      close all;
      for i=1:n_validations
        rng(i);
        % Cria um objeto split. Particiona a base em dois conjuntos
        % de classes treinadas e não treinadas. Separa uma
        % parte para treinamento e outra para teste
        obj.split{i}    = SplitData(obj.X,obj.Y,training_ratio,obj.untrained_classes);
        % Separa uma parte do treinamento para validação
        [id_train,id_val] = obj.split{i}.idTrainVal();
        [xtrain,ytrain,xval,yval] = obj.split{i}.dataTrainVal(id_train,id_val);
        RK = [];
        for j=1:obj.num_kernels
          fprintf('\nOne SVM \tVal: %d/%d \tKernel %d/%d',i,n_validations,j,obj.num_kernels);
          result = obj.evaluateOneSVM(xtrain,ytrain,xval,yval,obj.kernel(j));
          result.kernel = obj.kernel(j);
          mcc(i,j) = result.MCC;
          if plot_error
            RK = cat(1,RK,mcc(i,j));
            figure(1);
            clf('reset');
            plot(obj.kernel(1:j),RK,'-','LineWidth',3);
            xlim([obj.kernel(1),obj.kernel(end)]);
            ylim([0,1]);
            xlabel('Kernel');
            ylabel('Matthews correlation coefficient (MCC)');
            title(['One SVM [ validação ',num2str(i),'/',num2str(n_validations),' | kernel ',num2str(j),'/',num2str(obj.num_kernels),' ]']);
            drawnow;
            pause(0.01);
          end
        end
        model.split{i} = struct(obj.split{i});
      end
      mean_mcc = mean(mcc,1);
      [max_mean_mcc,id_max] = max(mean_mcc);
      
      model.training_ratio = training_ratio;
      model.kernel = obj.kernel(id_max);
      model.untrained_classes = obj.untrained_classes;
      model.mean_mcc = max_mean_mcc;
    end

    function model = validationMultiClass(obj,n_validations,plot_error)
      %-----------------------------------------------------------------------------------
      % This method runs validation experiments and hyperparameter search for MultiSVM.
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
      obj.split = cell(n_validations,1);
      mcc = zeros(obj.num_kernels,num_untrained_classes,n_validations);
      for i=1:n_validations
        rng(i);
        % Cria um objeto split. Particiona a base em dois conjuntos
        % de classes treinadas e não treinadas. Separa uma
        % parte para treinamento e outra para teste
        obj.split{i}    = SplitData(obj.X,obj.Y,training_ratio,obj.untrained_classes);
        % Separa uma parte do treinamento para validação
        [id_train,id_val] = obj.split{i}.idTrainVal();
        [xtrain,ytrain,xval,yval] = obj.split{i}.dataTrainVal(id_train,id_val);
        RK = [];
        for j=1:obj.num_kernels
          % Treina classificadores SVM's binários com a abordagem One vs All
          ctrain = unique(ytrain);
          model_svm = cell(numel(ctrain),1);
          for c=1:numel(ctrain)
            ytrain_binary = -ones(numel(ytrain),1);
            ytrain_binary(ytrain==ctrain(c)) = 1;
            if strcmp(obj.kernel_type,'poly')
              svm_model = fitcsvm(xtrain,ytrain_binary,'KernelFunction','polynomial', 'PolynomialOrder', obj.kernel(j));
            else
              svm_model = fitcsvm(xtrain,ytrain_binary,'KernelFunction','rbf', 'KernelScale', 1/(2*obj.kernel(j)^2) );
            end
            model_svm{c} = fitSVMPosterior(svm_model);
          end
          RT = [];
          for k=1:num_untrained_classes
            fprintf('\nMulti SVM \tVal: %d/%d \tKernel %d/%d \tDecision threshold %d/%d\n',i,n_validations,j,obj.num_kernels,k,num_untrained_classes);
            result = obj.evaluateMultiSVMAux(model_svm,ctrain,xval,yval,obj.kernel(j),hyperparameters.decision_thresholds(k));
            result.kernel = obj.kernel(j);
            mcc(j,k,i) = result.MCC;
            if plot_error
              RT = cat(1,RT,mcc(j,k,i));
              figure(1);
              pause(0.01);
              plot(hyperparameters.decision_thresholds(1:k),RT,'-r','LineWidth',3);
              xlim([0,1]);
              ylim([0,1]);
              xlabel('Threshold');
              ylabel('Matthews correlation coefficient (MCC)');
              title(['Multi SVM [ validação ',num2str(i),'/',num2str(n_validations),' | kernel ',num2str(j),'/',num2str(obj.num_kernels),' | decision_threshold ',num2str(k),'/',num2str(num_untrained_classes),' ]']);
              legend off;
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
            title(['Multi SVM [ validação ',num2str(i),'/',num2str(n_validations),' | kernel ',num2str(j),'/',num2str(obj.num_kernels),' ]']);
            drawnow;
          end
        end
        model.split{i} = struct(obj.split{i});
      end
      mean_mcc = mean(mcc,3);
      max_mean_mcc = max(max(mean_mcc));
      [id_k,id_t] = find(mean_mcc == max_mean_mcc);
      id_k = id_k(1); id_t = id_t(1);
      
      model.training_ratio = training_ratio;
      model.kernel = obj.kernel(id_k);
      model.decision_threshold = hyperparameters.decision_thresholds(id_t);
      model.untrained_classes = obj.untrained_classes;
      model.mean_mcc = max_mean_mcc;
    end
    
    function [results,evaluations] = evaluateOneSVMModel(obj,model,num_tests)
      % ----------------------------------------------------------------------------------
      % This method is used to evaluate the OneSVM prediction with multi-class novelty 
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
        fprintf('\nONE CLASS Test: %d/%d\n',i,num_tests);
        id_test = obj.split{i}.idTest();
        [xtest,ytest] = obj.split{i}.dataTest(id_test);
        [xtrain,ytrain] = obj.split{i}.dataTrain(obj.split{i}.id_train_val_t);
        evaluations{i} = obj.evaluateOneSVM(xtrain,ytrain,xtest,ytest,model.kernel);
      end
      results = struct2table(cell2mat(evaluations));
    end
    
    function [results,evaluations] = evaluateOneSVMTests(obj,xtrain,ytrain,xtest,ytest,model)
      % ----------------------------------------------------------------------------------
      % This method is used to evaluate the OneSVM prediction with multi-class novelty 
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
        fprintf('\nONE CLASS \tTest: %d/%d\n',i,num_tests);
        evaluations{i} = obj.evaluateOneSVM(xtrain,ytrain,xtest(:,:,i),ytest,model.kernel);
      end
      results = struct2table(cell2mat(evaluations));
    end
    
    function [results,evaluations] = evaluateMultiSVMModel(obj,model,num_tests)
      % ----------------------------------------------------------------------------------
      % This method is used to evaluate the MultiSVM prediction with multi-class novelty 
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
        fprintf('\nMULTI CLASS Test: %d/%d\n',i,num_tests);
        id_test = obj.split{i}.idTest();
        [xtest,ytest] = obj.split{i}.dataTest(id_test);
        [xtrain,ytrain] = obj.split{i}.dataTrain(obj.split{i}.id_train_val_t);
        evaluations{i} = obj.evaluateMultiSVM(xtrain,ytrain,xtest,ytest,model.kernel,model.decision_threshold);
      end
      results = struct2table(cell2mat(evaluations));
    end

    function [results,evaluations] = evaluateMultiSVMTests(obj,xtrain,ytrain,xtest,ytest,model)
      % ----------------------------------------------------------------------------------
      % This method is used to evaluate the MultiSVM prediction with multi-class novelty 
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
        fprintf('\nMULTI CLASS \tTest: %d/%d\n',i,num_tests);
        evaluations{i} = obj.evaluateMultiSVM(xtrain,ytrain,xtest(:,:,i),ytest,model.kernel,model.decision_threshold);
      end
      results = struct2table(cell2mat(evaluations));
    end
    
    function predictions = predictOneSVM(obj,xtrain,ytrain,xtest,kernel_arg)
      % ----------------------------------------------------------------------------------
      % This method is used to run OneSVM prediction with multi-class novelty detection.
      %
      % Input args
      %   xtrain: training data [num_train x dimensions].
      %   ytrain: training labels [num_train x 1].
      %   xtest: test data [num_test x dimensions].
      %   kernel_arg: kernel parameter.
      %
      % Output args:
      %   predictions: prediction with multi-class novelty detection.
      % ----------------------------------------------------------------------------------
      % Classes treinadas
      ctrain = unique(ytrain);
      
      % Treina One Class SVM's
      model = cell(numel(ctrain),1);
      scores = zeros(size(xtest,1),numel(ctrain));
      for i=1:numel(ctrain)
        xtrain_i =  xtrain(ytrain==ctrain(i),:);
        y = ones(size(xtrain_i,1),1);
        if strcmp(obj.kernel_type,'poly')
          model{i} = fitcsvm(xtrain_i,y,'KernelFunction','polynomial', 'PolynomialOrder', kernel_arg);
        else
          model{i} = fitcsvm(xtrain_i,y,'KernelFunction','rbf', 'KernelScale', 1/(2*kernel_arg^2) );
        end
        [~,score] = predict(model{i},xtest);
        scores(:,i) = score(:,1);
      end
      
      % Combina os classificadores para gerar uma classificação multi classe
      [max_scores,predictions_k] = max(scores,[],2);
      
      % Classifica outliers
      predictions_k(max_scores < 0) = -1;
      
      % Converte para a indexação das classes treinadas
      predictions = predictions_k;
      for i=1:numel(ctrain)
        predictions(predictions_k==i) = ctrain(i);
      end
    end
    
    function result = evaluateOneSVM(obj,xtrain,ytrain,xtest,ytest,kernel_arg)
      % ----------------------------------------------------------------------------------
      % This method is used to evaluate the OneSVM prediction with multi-class novelty 
      % detection.
      %
      % Input args
      %   xtrain: training data [num_train x dimensions].
      %   ytrain: training labels [num_train x 1].
      %   xtest: test data [num_test x dimensions].
      %   ytest: test labels [num_test x 1].
      %   kernel_arg: kernel parameter.
      %
      % Output args
      %   result: metrics report for multi-class prediction and novelty detection.
      % ----------------------------------------------------------------------------------

      % Classes treinadas
      ctrain = unique(ytrain);
      
      % Treina One Class SVM's
      model = cell(numel(ctrain),1);
      scores = zeros(numel(ytest),numel(ctrain));
      for i=1:numel(ctrain)
        xtrain_i =  xtrain(ytrain==ctrain(i),:);
        y = ones(size(xtrain_i,1),1);
        if strcmp(obj.kernel_type,'poly')
          model{i} = fitcsvm(xtrain_i,y,'KernelFunction','polynomial', 'PolynomialOrder', kernel_arg);
        else
          model{i} = fitcsvm(xtrain_i,y,'KernelFunction','rbf', 'KernelScale', 1/(2*kernel_arg^2));
        end
        [~,score] = predict(model{i},xtest);
        scores(:,i) = score(:,1);
      end
      
      % Combina os classificadores para gerar uma classificação multi classe
      [max_scores,predictions_k] = max(scores,[],2);
      
      % Classifica outliers
      predictions_k(max_scores < 0) = -1;
      
      % Converte para a indexação das classes treinadas
      predictions = predictions_k;
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
      
      fprintf('\n\nkernel: %f\nTPR: %f \nTNR: %f \nFPR: %f \nFNR: %f \nF1: %f \nMCC: %f \nACC: %f\nAFR: %f\n',...
        kernel_arg,report_outliers.TPR(2),report_outliers.TNR(2),report_outliers.FPR(2),report_outliers.FNR(2),report_outliers.F1(2),report_outliers.MCC(2),report_outliers.ACC(2),report_outliers.AFR(2));
    end

    function result = evaluateMultiSVM(obj,xtrain,ytrain,xtest,ytest,kernel_arg,decision_threshold)
      % ----------------------------------------------------------------------------------
      % This method is used to evaluate the MultiClass prediction with multi-class novelty 
      % detection.
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
      % Classes treinadas
      ctrain = unique(ytrain);
      
      % Treina classificadores SVM's binários com a abordagem One vs All
      model = cell(numel(ctrain),1);
      probability_score = zeros(numel(ytest),numel(ctrain));
      for i=1:numel(ctrain)
        ytrain_binary = -ones(numel(ytrain),1);
        ytrain_binary(ytrain==ctrain(i)) = 1;
        if strcmp(obj.kernel_type,'poly')
          svm_model = fitcsvm(xtrain,ytrain_binary,'KernelFunction','polynomial', 'PolynomialOrder', kernel_arg);
        else
          svm_model = fitcsvm(xtrain,ytrain_binary,'KernelFunction','rbf', 'KernelScale', 1/(2*kernel_arg^2) );
        end
        model{i} = fitSVMPosterior(svm_model);
        [~,score] = predict(model{i},xtest);
        probability_score(:,i) = score(:,2);
      end
      
      % Combina os classificadores binários para
      % gerar uma classificação multi classe
      [max_prob,predictions_k] = max(probability_score,[],2);
      
      % Classifica outliers
      predictions_k(max_prob < decision_threshold) = -1;
      
      % Converte para a indexação das classes treinadas
      predictions = predictions_k;
      for i=1:numel(ctrain)
        predictions(predictions_k==i) = ctrain(i);
      end
      
      % Report outliers
      outlier_gt = -ones(size(ytest));
      outlier_gt(ytest>0) = 1;
      
      outlier_predictions = -ones(size(predictions));
      outlier_predictions(predictions>0) = 1;
      
      report_outliers = ClassificationReport(outlier_gt,outlier_predictions);
      
      % Report
      report = ClassificationReport(ytest,predictions);
      
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

    function result = evaluateMultiSVMAux(obj,model,ctrain,xtest,ytest,kernel_arg,decision_threshold)
      % ----------------------------------------------------------------------------------
      % This method is used to evaluate the MultiClass prediction with multi-class novelty 
      % detection in validation experiments.
      %
      % Input args
      %   model: MultiSVM model.
      %   ctrain: trained classes
      %   xtest: test data [num_test x dimensions].
      %   ytest: test labels [num_test x 1].
      %   kernel_arg: kernel parameter.
      %   decision_threshold: decision threshold hyperparameter.
      %
      % Output args
      %   result: metrics report for multi-class prediction and novelty detection.
      % ----------------------------------------------------------------------------------

      probability_score = zeros(numel(ytest),numel(ctrain));
      for i=1:numel(ctrain)
        [~,score] = predict(model{i},xtest);
        probability_score(:,i) = score(:,2);
      end
      
      % Combina os classificadores binários para
      % gerar uma classificação multi classe
      [max_prob,predictions_k] = max(probability_score,[],2);
      
      % Classifica outliers
      predictions_k(max_prob < decision_threshold) = -1;
      
      % Converte para a indexação das classes treinadas
      predictions = predictions_k;
      for i=1:numel(ctrain)
        predictions(predictions_k==i) = ctrain(i);
      end
      
      % Report outliers
      outlier_gt = -ones(size(ytest));
      outlier_gt(ytest>0) = 1;
      
      outlier_predictions = -ones(size(predictions));
      outlier_predictions(predictions>0) = 1;
      
      report_outliers = ClassificationReport(outlier_gt,outlier_predictions);
      
      % Report
      report = ClassificationReport(ytest,predictions);
      
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
    
    function predictions = predictMultiSVM(obj,xtrain,ytrain,xtest,kernel_arg,decision_threshold)
      % ----------------------------------------------------------------------------------
      % This method is used to run MultiSVM prediction with multi-class novelty detection.
      %
      % Input args
      %   xtrain: training data [num_train x dimensions].
      %   ytrain: training labels [num_train x 1].
      %   xtest: test data [num_test x dimensions].
      %   kernel_arg: kernel parameter.
      %   decision_threshold: decision threshold hyperparameter.
      %
      % Output args:
      %   predictions: prediction with multi-class novelty detection.
      % ----------------------------------------------------------------------------------

      % Classes treinadas
      ctrain = unique(ytrain);
      
      % Treina classificadores SVM's binários com a abordagem One vs All
      model = cell(numel(ctrain),1);
      probability_score = zeros(size(xtest,1),numel(ctrain));
      for i=1:numel(ctrain)
        ytrain_binary = -ones(numel(ytrain),1);
        ytrain_binary(ytrain==ctrain(i)) = 1;
        if strcmp(obj.kernel_type,'poly')
          svm_model = fitcsvm(xtrain,ytrain_binary,'KernelFunction','polynomial', 'PolynomialOrder', kernel_arg);
        else
          svm_model = fitcsvm(xtrain,ytrain_binary,'KernelFunction','rbf', 'KernelScale', 1/(2*kernel_arg^2));
        end
        model{i} = fitSVMPosterior(svm_model);
        [~,score] = predict(model{i},xtest);
        probability_score(:,i) = score(:,2);
      end
      
      % Combina os classificadores binários para
      % gerar uma classificação multi classe
      [max_prob,predictions_k] = max(probability_score,[],2);
      
      % Classifica outliers
      predictions_k(max_prob < decision_threshold) = -1;
      
      % Converte para a indexação das classes treinadas
      predictions = predictions_k;
      for i=1:numel(ctrain)
        predictions(predictions_k==i) = ctrain(i);
      end
    end
  end
end
