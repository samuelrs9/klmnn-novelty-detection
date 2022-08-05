classdef KpcaND < handle
  % --------------------------------------------------------------------------------------
  % KPCA Novelty Detection for multi-class classification problems.
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
    decision_thresholds = [];% score thresholds list (the best needs to be found)
    training_ratio = 0;      % training sample rate
    split = {};              % holds a split object that helps the cross-validation process
    samples_per_classe = []; % samples per class
    kernel_type = [];        % kernel function type
    num_kernels = 0;         % number of kernel values
    kernel = [];             % kernel list for svm algorithm (the best must be found)        
  end
  
  methods
    function obj = KpcaND(X,Y,untrained_classes,training_ratio)
      % ----------------------------------------------------------------------------------
      % Constructor.
      %
      % Input args
      %   X: samples [num_samples x dimension].
      %   Y: sample labels [num_samples x 1].
      %   untrained_classes: number of untrained classes, this parameter can
      %     be used to simulate novelty data in the dataset.
      %   training_ratio: training sample rate.
      % ----------------------------------------------------------------------------------]
      obj.X = X;
      obj.Y = Y;
      obj.num_classes = numel(unique(Y));
      obj.training_ratio = 0.7;
      if nargin>=4
        obj.untrained_classes = untrained_classes;
        if nargin==5
          obj.training_ratio = training_ratio;
        end
      end
    end
    
    function experiment = runNoveltyDetectionExperiments(obj,num_experiments,plot_metric)
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
          RT = [];
          for k=1:obj.num_thresholds
            fprintf('\nKPCA Nov \tTest: %d/%d \tKernel (%d/%d) \tThreshold (%d/%d)\n',i,num_experiments,j,obj.num_kernels,k,obj.num_thresholds);
            decision_threshold = obj.decision_thresholds(k);
            evaluations{j,k,i} = obj.evaluate(xtrain,xtest,ytest,kernel_arg,decision_threshold);
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
              plot(obj.decision_thresholds(1:k),RT,'-r','LineWidth',3);
              xlim([obj.decision_thresholds(1),obj.decision_thresholds(end)]);
              ylim([0,1]);
              xlabel('Threshold');
              ylabel('Matthews correlation coefficient (MCC)');
              title(['KPCA Nov [ test ',num2str(i),'/',num2str(num_experiments),' | kernel ',num2str(j),'/',num2str(obj.num_kernels),' | decision_threshold ',num2str(k),'/',num2str(obj.num_thresholds),' ]']);
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
            title(['KPCA Nov [ test ',num2str(i),'/',num2str(num_experiments),' | kernel ',num2str(j),'/',num2str(obj.num_kernels),' ]']);
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
      model.decision_threshold = obj.decision_thresholds(best_threshold_id);
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
      
      figure; pcolor(obj.decision_thresholds,obj.kernel,mean_mcc); colorbar;
      xlabel('decision_threshold'); ylabel('kernel'); title('MCC');
      
      figure; pcolor(obj.decision_thresholds,obj.kernel,mean_afr); colorbar;
      xlabel('decision_threshold'); ylabel('kernel'); title('AFR');
    end
    
    function model = validation(obj,n_validations,plot_error)
      % ----------------------------------------------------------------------------------
      % Validação do algoritmo kpca out detection
      % ----------------------------------------------------------------------------------      
      obj.split = cell(n_validations,1);
      mcc = zeros(obj.num_kernels,obj.num_thresholds,n_validations);
      for i=1:n_validations
        rng(i);
        % Cria um objeto split. Particiona a base em dois conjuntos
        % de classes treinadas e não treinadas. Separa uma
        % parte para treinamento e outra para teste
        obj.split{i}    = SplitData(obj.X,obj.Y,obj.training_ratio,obj.untrained_classes);
        % Separa uma parte do treinamento para validação
        [id_train,id_val] = obj.split{i}.idTrainVal();
        [xtrain,ytrain,xval,yval] = obj.split{i}.dataTrainVal(id_train,id_val);
        RK = [];
        for j=1:obj.num_kernels
          kernel_arg = obj.kernel(j);
          RT = [];
          for k=1:obj.num_thresholds
            fprintf('\nKPCA \tVal: %d/%d \tKernel %d/%d \tThreshold %d/%d\n',i,n_validations,j,obj.num_kernels,k,obj.num_thresholds);
            decision_threshold = obj.decision_thresholds(k);
            result = obj.evaluate(xtrain,xval,yval,kernel_arg,decision_threshold);
            result.kernel = kernel_arg;
            mcc(j,k,i) = result.MCC;
            if plot_error
              RT = cat(1,RT,mcc(j,k,i));
              figure(1);
              clf('reset');
              plot(obj.decision_thresholds(1:k),RT,'-r','LineWidth',3);
              xlim([obj.decision_thresholds(1),obj.decision_thresholds(end)]);
              ylim([0,1]);
              xlabel('Threshold');
              ylabel('Matthews correlation coefficient (MCC)');
              title(['KPCA [ validação ',num2str(i),'/',num2str(n_validations),' | kernel ',num2str(j),'/',num2str(obj.num_kernels),' | decision_threshold ',num2str(k),'/',num2str(obj.num_thresholds),' ]']);
              drawnow;
              pause(0.01);
            end
          end
          if plot_error
            RK = cat(1,RK,max(RT));
            figure(2);
            clf('reset');
            pause(0.01);
            plot(obj.kernel(1:j),RK,'-','LineWidth',3);
            xlim([obj.kernel(1),obj.kernel(end)]);
            ylim([0,1]);
            xlabel('Kernel');
            ylabel('Matthews correlation coefficient (MCC)');
            title(['KPCA [ validação ',num2str(i),'/',num2str(n_validations),' | kernel ',num2str(j),'/',num2str(obj.num_kernels),' ]']);
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
      model.decision_threshold = obj.decision_thresholds(id_t);
      model.untrained_classes = obj.untrained_classes;
      model.mean_mcc = max_mean_mcc;
    end
    
    function [results,evaluations] = evaluateModel(obj,model,num_tests)
      % ----------------------------------------------------------------------------------
      % This method is used to evaluate the KPCA prediction with multi-class novelty 
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
        fprintf('\nKPCA NOV Test: %d/%d\n',i,num_tests);
        id_test = obj.split{i}.idTest();
        [xtest,ytest] = obj.split{i}.dataTest(id_test);
        [xtrain,~] = obj.split{i}.dataTrain(obj.split{i}.id_train_val_t);
        evaluations{i} = obj.evaluate(xtrain,xtest,ytest,model.kernel,model.decision_threshold);
      end
      results = struct2table(cell2mat(evaluations));
    end
    
    function [results,evaluations] = evaluateTests(obj,xtrain,xtest,ytest,model)
      % ----------------------------------------------------------------------------------
      % This method is used to evaluate the KPCA prediction with multi-class novelty 
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
        fprintf('\nKPCA NOV \tTest: %d/%d\n',i,num_tests);
        evaluations{i} = obj.evaluate(xtrain,xtest(:,:,i),ytest,model.kernel,model.decision_threshold);
      end
      results = struct2table(cell2mat(evaluations));
    end

    function result = evaluate(obj,xtrain,xtest,ytest,kernel_arg,decision_threshold)
      % ----------------------------------------------------------------------------------
      % This method is used to evaluate the KPCA prediction with multi-class novelty detection.
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
      % Predição
      outlier_predictions = obj.predictNovelty(xtrain,xtest,kernel_arg,decision_threshold);
      
      % Report outliers
      outlier_gt = -ones(size(ytest));
      outlier_gt(ytest>0) = 1;
      
      report_outliers = ClassificationReport(outlier_gt,outlier_predictions);
      
      result.kernel =  kernel_arg;
      result.decision_threshold =  decision_threshold;
      result.outlier_predictions = outlier_predictions;
      result.TPR = report_outliers.TPR(2);
      result.TNR = report_outliers.TNR(2);
      result.FPR = report_outliers.FPR(2);
      result.FNR = report_outliers.FNR(2);
      result.F1 = report_outliers.F1(2);
      result.MCC = report_outliers.MCC(2);
      result.ACC = report_outliers.ACC(2);
      result.AFR = report_outliers.AFR(2);
      result.outlier_conf_matrix = report_outliers.CM;
      
      fprintf('\n\nkernel: %f \nthreshold: %f \nTPR: %f \nTNR: %f \nFPR: %f \nFNR: %f \nF1: %f \nMCC: %f \nACC: %f\nAFR: %f\n',...
        kernel_arg,decision_threshold,report_outliers.TPR(2),report_outliers.TNR(2),report_outliers.FPR(2),report_outliers.FNR(2),report_outliers.F1(2),report_outliers.MCC(2),report_outliers.ACC(2),report_outliers.AFR(2));
    end
   
    function [predictions,errors] = predict(obj,xtrain,xtest,kernel_arg,decision_threshold)
      % ----------------------------------------------------------------------------------
      % This method is used to KPCA prediction with multi-class novelty detection.
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

      % Modelo
      model = obj.kpcaModel(xtrain,kernel_arg,decision_threshold);
      % Teste
      predictions = ones(size(xtest,1),1);
      errors = zeros(size(xtest,1),1);
      for i=1:size(xtest,1)
        errors(i,1) = obj.recerr(xtest(i,:),model.data,model.kernel,model.alpha,...
          model.alphaKrow,model.sumalpha,model.Ksum);
      end
      predictions(errors > model.maxerr) = -1;
    end
   
    function model = kpcaModel(obj,xtrain,kernel_arg,eig_rate)
      % ----------------------------------------------------------------------------------
      % This method is used to compute kernel pca model.
      %
      % Input args
      %   xtrain: training samples.
      %   kernel_arg: kernel parameter.
      %   eig_rate: eigvalue rate.
      %
      % Output args:
      %   model: kernel pca model.
      % ----------------------------------------------------------------------------------      
      [n,d] = size(xtrain);
      
      % computing kernel matrix K
      K = zeros(n,n);
      for i=1:n
        for j=i:n
          K(i,j) = obj.kernelH(xtrain(i,:),xtrain(j,:),kernel_arg);
          K(j,i) = K(i,j);
        end
      end
      
      % correct K for non-zero center of data in feature space:
      Krow = sum(K,1)/n;
      Ksum = sum(Krow)/n;
      for i=1:n
        for j=1:n
          K(i,j) = K(i,j) - Krow(i) - Krow(j) + Ksum;
        end
      end
      
      % extracting eigenvectors of K
      %[alpha,lambda] = eigs(K,n_eigenvalue,'lm',opts);
      
      % Código incluido --------------------------------------
      [alpha,lambda] = eig(K);
      
      lambda = real(diag(lambda));
      [~,index_sort] = sort(lambda,'descend');
      lambda = lambda(index_sort);
      alpha = alpha(:,index_sort);
      
      indices = cumsum(lambda/sum(lambda)) <= eig_rate;
      lambda = lambda(indices);
      alpha = alpha(:,indices);
      
      if strcmp(obj.kernel_type,'gauss') %|| strcmp(obj.kernel_type,'poly')
        indices = lambda > 1e-6;
        lambda = lambda(indices);
        alpha = alpha(:,indices);
      end
      
      % ------------------------------------------------------
      
      % normalize alpha:
      %alpha = alpha * inv(sqrt(lambda));
      alpha = alpha/sqrt(diag(lambda));
      
      % compute some helper vectors:
      sumalpha = sum(alpha,1);
      alphaKrow = Krow * alpha;
      
      % evaluating reconstruction error for all data points
      err = zeros(n,1);
      for i=1:n
        x = xtrain(i,:);
        err(i) = obj.recerr(x,xtrain,kernel_arg,alpha,alphaKrow,sumalpha,Ksum);
      end
      model.err = err;
      model.maxerr = max(err);
      model.kernel = kernel_arg;
      model.alpha = alpha;
      model.alphaKrow = alphaKrow;
      model.sumalpha = sumalpha;
      model.Ksum = Ksum;
      model.data = xtrain;
    end    
    
    function err = recerr(obj,x,xtrain,kernel_arg,alpha,alphaKrow,sumalpha,Ksum)
      % ----------------------------------------------------------------------------------
      % This function computes the reconstruction error of x in feature
      % space.
      %
      % Input args
      %   x: test sample.
      %   xtrain: train samples.
      %   kernel_arg:
      %   alpha: 
      %   alphaKrow:
      %   sumalpha: 
      %   Ksum:      
      %
      % Output args
      %   err: reconstruction error.
      % ----------------------------------------------------------------------------------      
      n = size(xtrain,1);
      k = zeros(1,n);
      for j=1:n
        k(j) = obj.kernelH(x,xtrain(j,:),kernel_arg);
      end
      % projections:
      f = k*alpha - sumalpha * (sum(k)/n - Ksum) - alphaKrow;
      % reconstruction error:
      err = obj.kernelH(x,x,kernel_arg) - 2*sum(k)/n + Ksum - f*f';
    end
    
    function k = kernelH(obj,x1,x2,kernel_arg)
      % ----------------------------------------------------------------------------------
      % this method is used to evaluate the kernel function with respect to
      % two samples x and y.
      %
      % Input args
      %   x1: sample 1.
      %   x2: sample 2.
      %   kernel_arg: kernel parameter.
      %
      % Output args:
      %   k: kernel function value for x1 and x2.
      % ----------------------------------------------------------------------------------      
      if strcmp(obj.kernel_type,'poly')
        offset = 1.0;
        k = (x1*x2' + offset).^kernel_arg;
      else
        gamma = 1/(2*kernel_arg^2);
        diff = x1-x2;
        k = exp(-(diff * diff')*gamma);
      end
    end
  end
end
