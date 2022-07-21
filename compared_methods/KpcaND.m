classdef KpcaND < handle
  properties
    X = [];                % Pontos X [n_points x dim]
    Y = [];                % Rótulos Y
    n_classes = 0;         % Números de classes
    untrained_classes = 0; % Número de classes não treinadas
    n_thresholds = 0;      % Número de thresholds de scores
    threshold = [];        % Vetor de thresholds de scores
    n_kernels = 0;         % Número de kernels para validação
    kernel_type = [];      % Tipo da função de kernel
    kernel = [];           % Vetor kernels (o melhor deve ser encontrado)
    training_ratio = 0;    % Taxa de treinamento de amostras
    split = {};            % Guarda um objeto split para auxiliar o processo de validação
  end
  
  methods
    function obj = KpcaND(X,Y,n_classes,untrained_classes,training_ratio)
      % ----------------------------------------------------------------------------------
      % Construtor
      % ----------------------------------------------------------------------------------      
      obj.X = X;
      obj.Y = Y;
      obj.n_classes = n_classes;
      obj.training_ratio = 0.7;
      if nargin>=4
        obj.untrained_classes = untrained_classes;
        if nargin==5
          obj.training_ratio = training_ratio;
        end
      end
    end
    
    function experiment = runNoveltyDetectionExperiments(obj,n_experiments,view_plot_metric)
      % ----------------------------------------------------------------------------------
      % Executa experimentos de detecção de novidade e busca de hiperparâmetros
      % ----------------------------------------------------------------------------------      
      split_exp = cell(n_experiments,1);
      
      MCC = zeros(obj.n_kernels,obj.n_thresholds,n_experiments);
      AFR = zeros(obj.n_kernels,obj.n_thresholds,n_experiments);
      F1 = zeros(obj.n_kernels,obj.n_thresholds,n_experiments);
      TPR = zeros(obj.n_kernels,obj.n_thresholds,n_experiments);
      TNR = zeros(obj.n_kernels,obj.n_thresholds,n_experiments);
      FPR = zeros(obj.n_kernels,obj.n_thresholds,n_experiments);
      FNR = zeros(obj.n_kernels,obj.n_thresholds,n_experiments);
      
      evaluations = cell(obj.n_kernels,obj.n_thresholds,n_experiments);
      
      for i=1:n_experiments
        rng(i);
        % Seleciona classes treinadas e não treinadas
        [trained,untrained,is_trained_class] = Split.selectClasses(obj.n_classes,obj.untrained_classes);
        
        % Divide os índices em treino e teste
        [idx_train,idx_test] = Split.trainTestIdx(obj.X,obj.Y,obj.training_ratio,obj.n_classes,is_trained_class);
        [xtrain,xtest,ytrain,ytest] = Split.dataTrainTest(idx_train,idx_test,obj.X,obj.Y);
        
        % Todas as amostras não treinadas são definidas
        % como outliers (label -1)
        ytest(logical(sum(ytest==untrained,2))) = -1;
        
        RK = [];
        for j=1:obj.n_kernels
          kernel_arg = obj.kernel(j);
          RT = [];
          for k=1:obj.n_thresholds
            fprintf('\nKPCA Nov \tTest: %d/%d \tKernel (%d/%d) \tThreshold (%d/%d)\n',i,n_experiments,j,obj.n_kernels,k,obj.n_thresholds);
            threshold_arg = obj.threshold(k);
            evaluations{j,k,i} = obj.evaluate(xtrain,xtest,ytest,kernel_arg,threshold_arg);
            evaluations{j,k,i}.kernel = kernel_arg;
            MCC(j,k,i) = evaluations{j,k,i}.MCC;
            F1(j,k,i) = evaluations{j,k,i}.F1;
            AFR(j,k,i) = evaluations{j,k,i}.AFR;
            TPR(j,k,i) = evaluations{j,k,i}.TPR;
            TNR(j,k,i) = evaluations{j,k,i}.TNR;
            FPR(j,k,i) = evaluations{j,k,i}.FPR;
            FNR(j,k,i) = evaluations{j,k,i}.FNR;
            if view_plot_metric
              RT = cat(1,RT,MCC(j,k,i));
              figure(1);
              clf('reset');
              plot(obj.threshold(1:k),RT,'-r','LineWidth',3);
              xlim([obj.threshold(1),obj.threshold(end)]);
              ylim([0,1]);
              xlabel('Threshold');
              ylabel('Matthews correlation coefficient (MCC)');
              title(['KPCA Nov [ test ',num2str(i),'/',num2str(n_experiments),' | kernel ',num2str(j),'/',num2str(obj.n_kernels),' | threshold ',num2str(k),'/',num2str(obj.n_thresholds),' ]']);
              drawnow;
              pause(0.01);
            end
          end
          if view_plot_metric
            RK = cat(1,RK,max(RT));
            figure(2);
            clf('reset');
            plot(obj.kernel(1:j),RK,'-','LineWidth',3);
            xlim([obj.kernel(1),obj.kernel(end)]);
            ylim([0,1]);
            xlabel('Kernel');
            ylabel('Matthews correlation coefficient (MCC)');
            title(['KPCA Nov [ test ',num2str(i),'/',num2str(n_experiments),' | kernel ',num2str(j),'/',num2str(obj.n_kernels),' ]']);
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
      
      figure; pcolor(obj.threshold,obj.kernel,mean_mcc); colorbar;
      xlabel('threshold'); ylabel('kernel'); title('MCC');
      
      figure; pcolor(obj.threshold,obj.kernel,mean_afr); colorbar;
      xlabel('threshold'); ylabel('kernel'); title('AFR');
    end
    
    function model = validation(obj,n_validations,view_plot_error)
      % ----------------------------------------------------------------------------------
      % Validação do algoritmo kpca out detection
      % ----------------------------------------------------------------------------------      
      obj.split = cell(n_validations,1);
      mcc = zeros(obj.n_kernels,obj.n_thresholds,n_validations);
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
        for j=1:obj.n_kernels
          kernel_arg = obj.kernel(j);
          RT = [];
          for k=1:obj.n_thresholds
            fprintf('\nKPCA \tVal: %d/%d \tKernel %d/%d \tThreshold %d/%d\n',i,n_validations,j,obj.n_kernels,k,obj.n_thresholds);
            threshold_arg = obj.threshold(k);
            result = obj.evaluate(xtrain,xval,yval,kernel_arg,threshold_arg);
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
              title(['KPCA [ validação ',num2str(i),'/',num2str(n_validations),' | kernel ',num2str(j),'/',num2str(obj.n_kernels),' | threshold ',num2str(k),'/',num2str(obj.n_thresholds),' ]']);
              drawnow;
              pause(0.01);
            end
          end
          if view_plot_error
            RK = cat(1,RK,max(RT));
            figure(2);
            clf('reset');
            pause(0.01);
            plot(obj.kernel(1:j),RK,'-','LineWidth',3);
            xlim([obj.kernel(1),obj.kernel(end)]);
            ylim([0,1]);
            xlabel('Kernel');
            ylabel('Matthews correlation coefficient (MCC)');
            title(['KPCA [ validação ',num2str(i),'/',num2str(n_validations),' | kernel ',num2str(j),'/',num2str(obj.n_kernels),' ]']);
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
        fprintf('\nKPCA NOV Test: %d/%d\n',i,n_tests);
        id_test = obj.split{i}.idTest();
        [xtest,ytest] = obj.split{i}.dataTest(id_test);
        [xtrain,~] = obj.split{i}.dataTrain(obj.split{i}.id_train_val_t);
        evaluations{i} = obj.evaluate(xtrain,xtest,ytest,model.kernel,model.threshold);
      end
      results = struct2table(cell2mat(evaluations));
    end
    
    function [results,evaluations] = evaluateTests(obj,xtrain,xtest,ytest,model)
      % ----------------------------------------------------------------------------------
      % Avalia o modelo treinado em conjuntos de testes
      % ----------------------------------------------------------------------------------      
      n_tests = size(xtest,3);
      evaluations = cell(n_tests,1);
      for i=1:n_tests
        fprintf('\nKPCA NOV \tTest: %d/%d\n',i,n_tests);
        evaluations{i} = obj.evaluate(xtrain,xtest(:,:,i),ytest,model.kernel,model.threshold);
      end
      results = struct2table(cell2mat(evaluations));
    end

    function result = evaluate(obj,xtrain,xtest,ytest,kernel_arg,threshold_arg)
      % ----------------------------------------------------------------------------------
      % Avalia o algoritmo kpca nov detection
      % ----------------------------------------------------------------------------------      
      % Predição
      outlier_predictions = obj.predictNovelty(xtrain,xtest,kernel_arg,threshold_arg);
      
      % Report outliers
      outlier_gt = -ones(size(ytest));
      outlier_gt(ytest>0) = 1;
      
      report_outliers = ClassificationReport(outlier_gt,outlier_predictions);
      
      result.kernel =  kernel_arg;
      result.threshold =  threshold_arg;
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
        kernel_arg,threshold_arg,report_outliers.TPR(2),report_outliers.TNR(2),report_outliers.FPR(2),report_outliers.FNR(2),report_outliers.F1(2),report_outliers.MCC(2),report_outliers.ACC(2),report_outliers.AFR(2));
    end
    
    function model = kpcaModel(obj,data,kernel_arg,eig_rate)
      % ----------------------------------------------------------------------------------
      % Calcula o modelo kpca
      % ----------------------------------------------------------------------------------      
      [n,d] = size(data);
      
      % computing kernel matrix K
      K = zeros(n,n);
      for i=1:n
        for j=i:n
          K(i,j) = obj.kernelH(data(i,:),data(j,:),kernel_arg);
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
        x = data(i,:);
        err(i) = obj.recerr(x,data,kernel_arg,alpha,alphaKrow,sumalpha,Ksum);
      end
      model.err = err;
      model.maxerr = max(err);
      model.kernel = kernel_arg;
      model.alpha = alpha;
      model.alphaKrow = alphaKrow;
      model.sumalpha = sumalpha;
      model.Ksum = Ksum;
      model.data = data;
    end
    
    function [predictions,errors] = predictNovelty(obj,xtrain,xtest,kernel_arg,threshold_arg)
      % ----------------------------------------------------------------------------------
      % This functions predict novelty
      % ----------------------------------------------------------------------------------      
      % Modelo
      model = obj.kpcaModel(xtrain,kernel_arg,threshold_arg);
      % Teste
      predictions = ones(size(xtest,1),1);
      errors = zeros(size(xtest,1),1);
      for i=1:size(xtest,1)
        errors(i,1) = obj.recerr(xtest(i,:),model.data,model.kernel,model.alpha,...
          model.alphaKrow,model.sumalpha,model.Ksum);
      end
      predictions(errors > model.maxerr) = -1;
    end
    
    function err = recerr(obj,x,data,kernel,alpha,alphaKrow,sumalpha,Ksum)
      % ----------------------------------------------------------------------------------
      % This function computes the reconstruction error of x in feature
      % space.
      % ----------------------------------------------------------------------------------      
      n = size(data,1);
      k = zeros(1,n);
      for j=1:n
        k(j) = obj.kernelH(x,data(j,:),kernel);
      end
      % projections:
      f = k*alpha - sumalpha * (sum(k)/n - Ksum) - alphaKrow;
      % reconstruction error:
      err = obj.kernelH(x,x,kernel) - 2*sum(k)/n + Ksum - f*f';
    end
    
    function k = kernelH(obj,x,y,kernel_arg)
      % ----------------------------------------------------------------------------------
      % Kernel function for kpca
      % ----------------------------------------------------------------------------------      
      % Código incluído -------------------------
      if strcmp(obj.kernel_type,'poly')
        offset = 1.0;
        k = (x*y' + offset).^kernel_arg;
      else
        % -----------------------------------------
        gamma = 1/(2*kernel_arg^2);
        diff = x-y;
        k = exp(-(diff * diff')*gamma);
      end
    end
  end
end
