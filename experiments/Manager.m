classdef Manager < handle
  
  properties
      X = [];
      y = [];
      method = [];
      parameters = [];
      out_dir = [];
      num_experiments = [];
      num_classes = [];
      untrained_classes = [];
      training_ratio = [];
      knn_arg = [];
      knn_threshold = [];
      plot_metric = [];    
  end
  
  methods
    function obj = Manager(X,y,method,parameters,out_dir,num_experiments,...
      num_classes,untrained_classes,training_ratio,knn_arg,knn_threshold,plot_metric)
      % ----------------------------------------------------------------------------------
      % Constructor.
      % ----------------------------------------------------------------------------------    
      obj.X = X;
      obj.y = y;
      obj.method = method;
      obj.parameters = parameters;
      obj.out_dir = out_dir;
      obj.num_experiments = num_experiments;
      obj.num_classes = num_classes;
      obj.untrained_classes = untrained_classes;
      obj.training_ratio = training_ratio;
      obj.knn_arg = knn_arg;
      obj.knn_threshold = knn_threshold;
      obj.plot_metric = plot_metric;
    end
    
    function runExperiments(obj)
      % ----------------------------------------------------------------------------------
      % Roda experimentos de validação e calibração de hiperparâmetros.
      % ----------------------------------------------------------------------------------
      knn_dir = strcat(obj.out_dir,'/K=',int2str(obj.knn_arg),' kappa=',int2str(obj.knn_threshold));
      if ~exist(knn_dir,'dir')
        mkdir(knn_dir);
      end
      for i=1:numel(obj.method)
        switch obj.method{i}
          case 'knn'
            try
              % Cria um objeto da classe KnnND
              knn = KnnND(obj.X,obj.y,knn_arg,knn_threshold,obj.num_classes,...
                obj.untrained_classes,obj.training_ratio);
              % Define intervalos de busca de parâmetros
              knn.num_thresholds = obj.parameters{1}.num_thresholds;
              knn.threshold = obj.parameters{1}.threshold;
              % Inicia experimentos              
              t0_knn = tic;              
              experiments = knn.runExperiments(obj.num_experiments,obj.plot_metric);              
              experiments.experiment_time = toc(t0_knn);
              experiments.num_validations = obj.num_experiments;
              experiments.search_thresholds = knn.threshold;
              experiments.num_search_parameters = knn.num_thresholds;
              % Salva experimentos
              save(strcat(knn_dir,'/knn_experiments.mat'),'-struct','experiments');
            catch
              fprintf('\n knn experiment error!!! \n');
            end
          case 'lmnn'
            try
              % Cria um objeto da classe LmnnND
              lmnn = LmnnND(obj.X,obj.y,knn_arg,knn_threshold,obj.num_classes,...
                obj.untrained_classes,obj.training_ratio);
              % Define intervalos de busca de parâmetros
              lmnn.num_thresholds = obj.parameters{2}.num_thresholds;
              lmnn.threshold = obj.parameters{2}.threshold;
              % Inicia experimentos
              t0_lmnn = tic;
              experiments = lmnn.runExperiments(obj.num_experiments,obj.plot_metric);
              experiments.experiment_time = toc(t0_lmnn);
              experiments.obj.num_experiments = obj.num_experiments;
              experiments.search_thresholds = lmnn.threshold;
              experiments.num_search_parameters = lmnn.num_thresholds;
              % Salva experimentos
              save(strcat(knn_dir,'/lmnn_experiments.mat'),'-struct','experiments');
            catch
              fprintf('\n lmnn experiment error!!! \n');
            end
          case 'klmnn'
            try
              % Cria um objeto da classe KlmnnND
              klmnn = KlmnnND(obj.X,obj.y,knn_arg,knn_threshold,obj.num_classes,...
                obj.untrained_classes,obj.training_ratio);
              % Define intervalos de busca de parâmetros
              klmnn.num_thresholds = obj.parameters{3}.num_thresholds;
              klmnn.threshold = obj.parameters{3}.threshold;
              klmnn.kernel_type = obj.parameters{3}.kernel_type;
              klmnn.num_kernels = obj.parameters{3}.num_kernels;
              klmnn.kernel = obj.parameters{3}.kernel;
              % Inicia experimentos
              t0_klmnn = tic;
              experiments = klmnn.runExperiments(obj.num_experiments,obj.plot_metric);
              experiments.experiment_time = toc(t0_klmnn);
              experiments.obj.num_experiments = obj.num_experiments;
              experiments.search_thresholds = klmnn.threshold;
              experiments.num_search_parameters = klmnn.num_thresholds;
              experiments.kernel_type = klmnn.kernel_type;
              experiments.search_kernels = klmnn.kernel;
              experiments.num_search_parameters = klmnn.num_thresholds * klmnn.num_kernels;
              % Salva experimentos
              save(strcat(knn_dir,'/klmnn_experiments.mat'),'-struct','experiments');
            catch
              fprintf('\n klmnn experiment error!!! \n');
            end
          case 'knfst'
            try
              % Cria um objeto da classe KnfstND
              knfst = KnfstND(obj.X,obj.y,obj.num_classes,obj.untrained_classes,obj.training_ratio);
              % Define intervalos de busca de parâmetros
              knfst.num_thresholds = obj.parameters{4}.num_thresholds;
              knfst.threshold = obj.parameters{4}.threshold;
              knfst.kernel_type = obj.parameters{4}.kernel_type;
              knfst.num_kernels = obj.parameters{4}.num_kernels;
              knfst.kernel = obj.parameters{4}.kernel;
              % Inicia experimentos
              t0_knfst = tic;
              experiments = knfst.runExperiments(obj.num_experiments,obj.plot_metric);
              experiments.experiment_time = toc(t0_knfst);
              experiments.obj.num_experiments = obj.num_experiments;
              experiments.search_thresholds = knfst.threshold;
              experiments.num_search_parameters = knfst.num_thresholds;
              experiments.kernel_type = knfst.kernel_type;
              experiments.search_kernels = knfst.kernel;
              experiments.num_search_parameters = knfst.num_thresholds * knfst.num_kernels;
              % Salva experimentos
              save(strcat(obj.out_dir,'/knfst_experiments.mat'),'-struct','experiments');
            catch
              fprintf('\n knfst experiment error!!! \n');
            end
          case 'one_svm'
            try
              % Cria um objeto da classe SvmND
              one_svm = SvmND(obj.X,obj.y,obj.num_classes,obj.untrained_classes,obj.training_ratio);
              % Define intervalos de busca de parâmetros
              one_svm.kernel_type = obj.parameters{5}.kernel_type;
              one_svm.num_kernels = obj.parameters{5}.num_kernels;
              one_svm.kernel = obj.parameters{5}.kernel;
              % Inicia experimentos
              t0_one_svm = tic;
              experiments = one_svm.runNoveltyDetectionOneSVMExperiments(...
                obj.num_experiments,obj.plot_metric);
              experiments.experiment_time = toc(t0_one_svm);
              experiments.obj.num_experiments = obj.num_experiments;
              experiments.kernel_type = one_svm.kernel_type;
              experiments.search_kernels = one_svm.kernel;
              experiments.num_search_parameters = one_svm.num_kernels;
              % Salva experimentos
              save(strcat(obj.out_dir,'/one_svm_experiments.mat'),'-struct','experiments');
            catch
              fprintf('\n one svm experiment error!!! \n');
            end
          case 'multi_svm'
            try
              % Cria um objeto da classe SvmND
              multi_svm = SvmND(obj.X,obj.y,obj.num_classes,obj.untrained_classes,obj.training_ratio);
              % Define intervalos de busca de parâmetros
              multi_svm.num_thresholds = obj.parameters{6}.num_thresholds;
              multi_svm.threshold = obj.parameters{6}.threshold;
              multi_svm.kernel_type = obj.parameters{6}.kernel_type;
              multi_svm.num_kernels = obj.parameters{6}.num_kernels;
              multi_svm.kernel = obj.parameters{6}.kernel;
              % Inicia experimentos
              t0_multi_svm = tic;
              experiments = multi_svm.runNoveltyDetectionMultiSVMExperiments(...
                obj.num_experiments,obj.plot_metric);
              experiments.experiment_time = toc(t0_multi_svm);
              experiments.obj.num_experiments = obj.num_experiments;
              experiments.search_thresholds = multi_svm.threshold;
              experiments.kernel_type = multi_svm.kernel_type;
              experiments.search_kernels = multi_svm.kernel;
              experiments.num_search_parameters = ...
                multi_svm.num_thresholds * multi_svm.num_kernels;
              % Salva experimentos
              save(strcat(obj.out_dir,'/multi_svm_experiments.mat'),'-struct','experiments');
            catch
              fprintf('\n multi svm experiment error!!! \n');
            end
          case 'kpca'
            try
              % Cria um objeto da classe KpcaND
              kpca = KpcaND(obj.X,obj.y,obj.num_classes,obj.untrained_classes,obj.training_ratio);
              % Define intervalos de busca de parâmetros
              kpca.num_thresholds = obj.parameters{7}.num_thresholds;
              kpca.threshold = obj.parameters{7}.threshold;
              kpca.kernel_type = obj.parameters{7}.kernel_type;
              kpca.num_kernels = obj.parameters{7}.num_kernels;
              kpca.kernel = obj.parameters{7}.kernel;
              % Inicia experimentos
              t0_kpca = tic;
              experiments = kpca.runExperiments(obj.num_experiments,obj.plot_metric);
              experiments.validation_time = toc(t0_kpca);
              experiments.obj.num_experiments = obj.num_experiments;
              experiments.search_thresholds = kpca.threshold;
              experiments.kernel_type = kpca.kernel_type;
              experiments.search_kernels = kpca.kernel;
              experiments.num_search_parameters = kpca.num_thresholds * kpca.num_kernels;
              % Salva experimentos
              save(strcat(obj.out_dir,'/kpca_experiments.mat'),'-struct','experiments');
            catch
              fprintf('\n kpca experiment error!!! \n');
            end
        end
      end
    end
    
    function runExperimentsForKnnMethods(obj)
      % ----------------------------------------------------------------------------------
      % Calibra os parâmetros do knn, lmnn e klmnn para diferentes valores de K e kappa
      % ----------------------------------------------------------------------------------      
      for K = 1:5
        for kappa = K:-1:K-3
          if kappa >= 1
            fprintf('\nK = %d \tkappa = %d\n',K,kappa);
            Manager.runExperiments(obj.X,obj.y,obj.method,obj.parameters,obj.out_dir,num_experiments,obj.num_classes,...
              obj.untrained_classes,obj.training_ratio,K,kappa,obj.plot_metric);
          end
        end
      end
    end
    
    function runEvaluationModels(obj)
      % ----------------------------------------------------------------------------------
      % Avalia os parâmetros ótimos usando uma parte das amostras não
      % utilizadas na validação
      % ----------------------------------------------------------------------------------      
      knn_dir = strcat(model_dir,'/K=',int2str(obj.knn_arg),' kappa=',int2str(obj.knn_threshold));
      if ~exist(knn_dir,'dir')
        mkdir(knn_dir);
      end
      % Testa os modelos
      for i=1:numel(obj.method)
        switch obj.method{i}
          case 'knn'
            try
              fprintf('\n-> KNN Novelty Detection \n');
              % Carrega o modelo
              knn_model = load(strcat(knn_dir,'/knn_model.mat'));
              knn = knn_model.knn;
              knn_model = knn_model.knn_model;
              % Inicia avaliação
              t0_knn = tic;
              [knn_evaluations.results,knn_evaluations.evaluations] = ...
                knn.evaluateModel(knn_model,num_experiments);-+0
              knn_evaluations.model = knn_model;
              knn_evaluations.evaluation_time = toc(t0_knn);
              % Salva avaliação
              save(strcat(knn_dir,'/knn_evaluations.mat'),'knn_evaluations');
            catch
              fprintf('\n error!!! \n');
            end
          case 'lmnn'
            try
              fprintf('\n-> LMNN Novelty Detection \n');
              % Carrega o modelo
              lmnn_model = load(strcat(knn_dir,'/lmnn_model.mat'));
              lmnn = lmnn_model.lmnn;
              lmnn_model = lmnn_model.lmnn_model;
              % Inicia avaliação
              t0_lmnn = tic;
              [lmnn_evaluations.results,lmnn_evaluations.evaluations] = ...
                lmnn.evaluateModel(lmnn_model,num_experiments);
              lmnn_evaluations.model = lmnn_model;
              lmnn_evaluations.evaluation_time = toc(t0_lmnn);
              % Salva avaliação
              save(strcat(knn_dir,'/lmnn_evaluations.mat'),'lmnn_evaluations');
            catch
              fprintf('\n error!!! \n');
            end
          case 'klmnn'
            try
              fprintf('\n-> KLMNN Novelty Detection \n');
              % Carrega o modelo
              klmnn_model = load(strcat(knn_dir,'/klmnn_model.mat'));
              klmnn = klmnn_model.klmnn;
              klmnn_model = klmnn_model.klmnn_model;
              % Inicia avaliação
              t0_klmnn = tic;
              [klmnn_evaluations.results,klmnn_evaluations.evaluations] = ...
                klmnn.evaluateModel(klmnn_model,num_experiments);
              klmnn_evaluations.model = klmnn_model;
              klmnn_evaluations.evaluation_time = toc(t0_klmnn);
              % Salva avaliação
              save(strcat(knn_dir,'/klmnn_evaluations.mat'),'klmnn_evaluations');
            catch
              fprintf('\n error!!! \n');
            end
          case 'knfst'
            try
              fprintf('\n-> KNFST Novelty Detection \n');
              % Carrega o modelo
              knfst_model = load(strcat(model_dir,'/knfst_model.mat'));
              knfst = knfst_model.knfst;
              knfst_model = knfst_model.knfst_model;
              % Inicia avaliação
              t0_knfst = tic;
              [knfst_evaluations.results,knfst_evaluations.evaluations] = ...
                knfst.evaluateModel(knfst_model,num_experiments);
              knfst_evaluations.model = knfst_model;
              knfst_evaluations.evaluation_time = toc(t0_knfst);
              % Salva avaliação
              save(strcat(model_dir,'/knfst_evaluations.mat'),'knfst_evaluations');
            catch
              fprintf('\n error!!! \n');
            end
          case 'one_svm'
            try
              fprintf('\n-> One SVM Novelty Detection \n');
              % Carrega o modelo
              one_svm_model = load(strcat(model_dir,'/one_svm_model.mat'));
              one_svm = one_svm_model.one_svm;
              one_svm_model = one_svm_model.one_svm_model;
              % Inicia avaliação
              t0_one_svm = tic;
              [one_svm_evaluations.results,one_svm_evaluations.evaluations] = ...
                one_svm.evaluateOneClassModel(one_svm_model,num_experiments);
              one_svm_evaluations.model = one_svm_model;
              one_svm_evaluations.evaluation_time = toc(t0_one_svm);
              % Salva avaliação
              save(strcat(model_dir,'/one_svm_evaluations.mat'),'one_svm_evaluations');
            catch
              fprintf('\n error!!! \n');
            end
          case 'multi_svm'
            try
              fprintf('\n-> Multi SVM Novelty Detection \n');
              % Carrega o modelo
              multi_svm_model = load(strcat(model_dir,'/multi_svm_model.mat'));
              multi_svm = multi_svm_model.multi_svm;
              multi_svm_model = multi_svm_model.multi_svm_model;
              % Inicia avaliação
              t0_multi_svm = tic;
              [multi_svm_evaluations.results,multi_svm_evaluations.evaluations] = ...
                multi_svm.evaluateMultiClassModel(multi_svm_model,num_experiments);
              multi_svm_evaluations.model = multi_svm_model;
              multi_svm_evaluations.evaluation_time = toc(t0_multi_svm);
              % Salva avaliação
              save(strcat(model_dir,'/multi_svm_evaluations.mat'),'multi_svm_evaluations');
            catch
              fprintf('\n error!!! \n');
            end
          case 'kpca'
            try
              fprintf('\n-> KPCA Novelty Detection \n');
              % Carrega o modelo
              kpca_model = load(strcat(model_dir,'/kpca_model.mat'));
              kpca = kpca_model.kpca;
              kpca_model = kpca_model.kpca_model;
              % Inicia avaliação
              t0_kpca = tic;
              [kpca_evaluations.results,kpca_evaluations.evaluations] = ...
                kpca.evaluateModel(kpca_model,num_experiments);
              kpca_evaluations.model = kpca_model;
              kpca_evaluations.evaluation_time = toc(t0_kpca);
              % Salva avaliação
              save(strcat(model_dir,'/kpca_evaluations.mat'),'kpca_evaluations');
            catch
              fprintf('\n error!!! \n');
            end
        end
      end
    end
    
    function runEvaluationTests(obj,xtrain,ytrain,xtest,ytest,model_dir)
      % ----------------------------------------------------------------------------------
      % Avalia os parâmetros ótimos em conjuntos de testes
      % ----------------------------------------------------------------------------------    
      knn_dir = strcat(model_dir,'/K=',int2str(obj.knn_arg),' kappa=',int2str(obj.knn_threshold));
      if ~exist(knn_dir,'dir')
        mkdir(knn_dir);
      end
      % Testa os modelos
      for i=1:numel(obj.method)
        switch obj.method{i}
          case 'knn'
            try
              fprintf('\n-> KNN Novelty Detection \n');
              % Carrega os parâmetros
              knn_model = load(strcat(knn_dir,'/knn_model.mat'));
              knn_model = knn_model.knn_model;
              % Inicia avaliação
              t0_knn = tic;
              knn = KnnND(obj,xtrain,ytrain,knn_model.obj.knn_arg,...
                knn_model.obj.knn_threshold,obj.num_classes);
              [knn_evaluations.results,knn_evaluations.evaluations] = ...
                knn.evaluateTests(obj,xtrain,ytrain,xtest,ytest,knn_model);
              knn_evaluations.model = knn_model;
              knn_evaluations.evaluation_time = toc(t0_knn);
              % Salva avaliação
              save(strcat(knn_dir,'/knn_evaluation_tests.mat'),'-struct','knn_evaluations');
            catch
              fprintf('\n--> knn evaluation error!!! \n');
            end
          case 'lmnn'
            try
              fprintf('\n-> LMNN Novelty Detection \n');
              % Carrega o modelo
              lmnn_model = load(strcat(knn_dir,'/lmnn_model.mat'));
              lmnn_model = lmnn_model.lmnn_model;
              % Inicia avaliação
              t0_lmnn = tic;
              lmnn = LmnnND(obj,xtrain,ytrain,lmnn_model.obj.knn_arg,...
                lmnn_model.obj.knn_threshold,obj.num_classes);
              [lmnn_evaluations.results,lmnn_evaluations.evaluations] = ...
                lmnn.evaluateTests(obj,xtrain,ytrain,xtest,ytest,lmnn_model);
              lmnn_evaluations.model = lmnn_model;
              lmnn_evaluations.evaluation_time = toc(t0_lmnn);
              % Salva avaliação
              save(strcat(knn_dir,'/lmnn_evaluation_tests.mat'),...
                '-struct','lmnn_evaluations');
            catch
              fprintf('\n--> lmnn evaluation error!!! \n');
            end
          case 'klmnn'
            try
              fprintf('\n-> KLMNN Novelty Detection \n');
              % Carrega o modelo
              klmnn_model = load(strcat(knn_dir,'/klmnn_model.mat'));
              klmnn_model = klmnn_model.klmnn_model;
              % Inicia avaliação
              t0_klmnn = tic;
              klmnn = KlmnnND(obj,xtrain,ytrain,klmnn_model.obj.knn_arg,...
                klmnn_model.obj.knn_threshold,obj.num_classes);
              [klmnn_evaluations.results,klmnn_evaluations.evaluations] = ...
                klmnn.evaluateTests(obj,xtrain,ytrain,xtest,ytest,klmnn_model);
              klmnn_evaluations.model = klmnn_model;
              klmnn_evaluations.evaluation_time = toc(t0_klmnn);
              % Salva avaliação
              save(strcat(knn_dir,'/klmnn_evaluation_tests.mat'),...
                '-struct','klmnn_evaluations');
            catch
              fprintf('\n--> klmnn evaluation error!!! \n');
            end
          case 'knfst'
            try
              fprintf('\n-> KNFST Novelty Detection \n');
              % Carrega o modelo
              knfst_model = load(strcat(model_dir,'/knfst_model.mat'));
              knfst_model = knfst_model.knfst_model;
              % Inicia avaliação
              t0_knfst = tic;
              knfst = KnfstND(obj,xtrain,ytrain,obj.num_classes);
              [knfst_evaluations.results,knfst_evaluations.evaluations] = ...
                knfst.evaluateTests(obj,xtrain,ytrain,xtest,ytest,knfst_model);
              knfst_evaluations.model = knfst_model;
              knfst_evaluations.evaluation_time = toc(t0_knfst);
              % Salva avaliação
              save(strcat(model_dir,'/knfst_evaluation_tests.mat'),...
                '-struct','knfst_evaluations');
            catch
              fprintf('\n--> knfst evaluation error!!! \n');
            end
          case 'one_svm'
            try
              fprintf('\n-> One SVM Novelty Detection \n');
              % Carrega o modelo
              one_svm_model = load(strcat(model_dir,'/one_svm_model.mat'));
              one_svm_model = one_svm_model.one_svm_model;
              % Inicia avaliação
              t0_one_svm = tic;
              one_svm = SvmND(obj,xtrain,ytrain,obj.num_classes);
              [one_svm_evaluations.results,one_svm_evaluations.evaluations] = ...
                one_svm.evaluateOneClassTests(obj,xtrain,ytrain,xtest,ytest,one_svm_model);
              one_svm_evaluations.model = one_svm_model;
              one_svm_evaluations.evaluation_time = toc(t0_one_svm);
              % Salva avaliação
              save(strcat(model_dir,'/one_svm_evaluation_tests.mat'),...
                '-struct','one_svm_evaluations');
            catch
              fprintf('\n--> one svm evaluation error!!! \n');
            end
          case 'multi_svm'
            try
              fprintf('\n-> Multi SVM Novelty Detection \n');
              % Carrega o modelo
              multi_svm_model = load(strcat(model_dir,'/multi_svm_model.mat'));
              multi_svm_model = multi_svm_model.multi_svm_model;
              % Inicia avaliação
              t0_multi_svm = tic;
              multi_svm = SvmND(obj,xtrain,ytrain,obj.num_classes);
              [multi_svm_evaluations.results,multi_svm_evaluations.evaluations] = ...
                multi_svm.evaluateMultiClassTests(obj,xtrain,ytrain,xtest,ytest,multi_svm_model);
              multi_svm_evaluations.model = multi_svm_model;
              multi_svm_evaluations.evaluation_time = toc(t0_multi_svm);
              % Salva avaliação
              save(strcat(model_dir,'/multi_svm_evaluation_tests.mat'),...
                '-struct','multi_svm_evaluations');
            catch
              fprintf('\n--> multi svm evaluation error!!! \n');
            end
          case 'kpca'
            try
              fprintf('\n-> KPCA Novelty Detection \n');
              % Carrega o modelo
              kpca_model = load(strcat(model_dir,'/kpca_model.mat'));
              kpca_model = kpca_model.kpca_model;
              % Inicia avaliação
              t0_kpca = tic;
              kpca = KpcaND(obj,xtrain,ytrain,obj.num_classes);
              [kpca_evaluations.results,kpca_evaluations.evaluations] = ...
                kpca.evaluateTests(obj,xtrain,xtest,ytest,kpca_model);
              kpca_evaluations.model = kpca_model;
              kpca_evaluations.evaluation_time = toc(t0_kpca);
              % Salva avaliação
              save(strcat(model_dir,'/kpca_evaluation_tests.mat'),...
                '-struct','kpca_evaluations');
            catch
              fprintf('\n--> kpca nov evaluation error!!! \n');
            end
        end
      end
    end
    
    function runEvaluationModelsForKnnMethods(obj,model_dir,num_experiments)
      % ----------------------------------------------------------------------------------
      % Avalia os modelos dos métodos knn, lmnn e klmnn usando uma parte das amostras não
      % utilizadas na validação
      % ----------------------------------------------------------------------------------      
      % Testes
      for K = 1:5
        for kappa = K:-1:K-3
          if kappa >= 1
            fprintf('\nK = %d \tkappa = %d\n',K,kappa);
            Manager.runEvaluationModels(obj.method,model_dir,num_experiments,K,kappa);
          end
        end
      end
    end
    
    function runEvaluations(obj,xtrain,ytrain,xtest,ytest,model_dir)
      % ----------------------------------------------------------------------------------
      % Avalia os métodos em um conjunto de testes com os parâmetros ótimos.
      % ----------------------------------------------------------------------------------    
      knn_dir = strcat(model_dir,'/K=',int2str(obj.knn_arg),' kappa=',int2str(obj.knn_threshold));
      if ~exist(knn_dir,'dir')
        mkdir(knn_dir);
      end
      for i=1:numel(obj.method)
        switch obj.method{i}
          case 'knn'
            fprintf('\n-> KNN Novelty Detection\n\n');
            % Carrega o modelo
            knn_model = load(strcat(knn_dir,'/knn_model.mat'));
            knn_model = knn_model.knn_model;
            % Avalia o modelo
            t0_knn = tic;
            knn = KnnND(obj,xtrain,ytrain,knn_model.obj.knn_arg,knn_model.obj.knn_threshold,obj.num_classes);
            knn_evaluation = knn.evaluate(obj,xtrain,ytrain,xtest,ytest,knn_model.threshold);
            knn_evaluation.evaluation_time = toc(t0_knn);
            % Salva avaliação
            save(strcat(knn_dir,'/knn_evaluation.mat'),'-struct','knn_evaluation');
          case 'lmnn'
            fprintf('\n-> LMNN Novelty Detection\n');
            % Carrega o modelo
            lmnn_model = load(strcat(knn_dir,'/lmnn_model.mat'));
            lmnn_model = lmnn_model.lmnn_model;
            % Avalia o modelo
            t0_lmnn = tic;
            lmnn = LmnnND(obj,xtrain,ytrain,lmnn_model.obj.knn_arg,...
              lmnn_model.obj.knn_threshold,obj.num_classes);
            lmnn_evaluation = lmnn.evaluate(obj,xtrain,ytrain,xtest,ytest,lmnn_model.threshold);
            lmnn_evaluation.evaluation_time = toc(t0_lmnn);
            % Salva avaliação
            save(strcat(knn_dir,'/lmnn_evaluation.mat'),'-struct','lmnn_evaluation');
          case 'klmnn'
            fprintf('\n-> KLMNN Novelty Detection\n');
            % Carrega o modelo
            klmnn_model = load(strcat(knn_dir,'/klmnn_model.mat'));
            klmnn_model = klmnn_model.klmnn_model;
            % Avalia o modelo
            t0_klmnn = tic;
            klmnn = KlmnnND(obj,xtrain,ytrain,klmnn_model.obj.knn_arg,...
              klmnn_model.obj.knn_threshold,obj.num_classes);
            klmnn.kernel_type = klmnn_model.kernel_type;
            klmnn_evaluation = klmnn.evaluate(obj,xtrain,ytrain,xtest,ytest,...
              klmnn_model.kernel,klmnn_model.threshold);
            klmnn_evaluation.evaluation_time = toc(t0_klmnn);
            % Salva avaliação
            save(strcat(knn_dir,'/klmnn_evaluation.mat'),'-struct','klmnn_evaluation');
          case 'knfst'
            fprintf('\n-> KNFST Novelty Detection\n');
            % Carrega o modelo
            knfst_model = load(strcat(model_dir,'/knfst_model.mat'));
            knfst_model = knfst_model.knfst_model;
            % Avalia o modelo
            t0_knfst = tic;
            knfst = KnfstND(obj,xtrain,ytrain,obj.num_classes);
            knfst.kernel_type = knfst_model.kernel_type;
            knfst_evaluation = knfst.evaluate(obj,xtrain,ytrain,xtest,ytest,...
              knfst_model.kernel,knfst_model.threshold);
            knfst_evaluation.evaluation_time = toc(t0_knfst);
            % Salva avaliação
            save(strcat(model_dir,'/knfst_evaluation.mat'),'-struct','knfst_evaluation');
          case 'one_svm'
            fprintf('\n-> One SVM Novelty Detection\n');
            % Carrega o modelo
            one_svm_model = load(strcat(model_dir,'/one_svm_model.mat'));
            one_svm_model = one_svm_model.one_svm_model;
            % Avalia o modelo
            t0_svm = tic;
            one_svm = SvmND(obj,xtrain,ytrain,obj.num_classes);
            one_svm.kernel_type = one_svm_model.kernel_type;
            one_svm_evaluation = one_svm.evaluateOneClassSVM(obj,xtrain,ytrain,...
              xtest,ytest,one_svm_model.kernel);
            one_svm_evaluation.evaluation_time = toc(t0_svm);
            % Salva avaliação
            save(strcat(model_dir,'/one_svm_evaluation.mat'),...
              '-struct','one_svm_evaluation');
          case 'multi_svm'
            fprintf('\n-> Multi SVM Novelty Detection\n');
            % Carrega o modelo
            multi_svm_model = load(strcat(model_dir,'/multi_svm_model.mat'));
            multi_svm_model = multi_svm_model.multi_svm_model;
            % Avalia o modelo
            t0_svm = tic;
            multi_svm = SvmND(obj,xtrain,ytrain,obj.num_classes);
            multi_svm.kernel_type = multi_svm_model.kernel_type;
            multi_svm_evaluation = multi_svm.evaluateMultiClassSVM(obj,xtrain,ytrain,...
              xtest,ytest,multi_svm_model.kernel,multi_svm_model.threshold);
            multi_svm_evaluation.evaluation_time = toc(t0_svm);
            % Salva avaliação
            save(strcat(model_dir,'/multi_svm_evaluation.mat'),...
              '-struct','multi_svm_evaluation');
          case 'kpca'
            fprintf('\n-> KPCA Novelty Detection\n');
            % Carrega o modelo
            kpca_model = load(strcat(model_dir,'/kpca_model.mat'));
            kpca_model = kpca_model.kpca_model;
            % Avalia o modelo
            t0_kpca = tic;
            kpca = KpcaND(obj,xtrain,ytrain,obj.num_classes);
            kpca.kernel_type = kpca_model.kernel_type;
            kpca_evaluation = kpca.evaluate(obj,xtrain,xtest,ytest,...
              kpca_model.kernel,kpca_model.threshold);
            kpca_evaluation.evaluation_time = toc(t0_kpca);
            % Salva avaliação
            save(strcat(model_dir,'/kpca_evaluation.mat'),'-struct','kpca_evaluation');
        end
      end
    end
    
    function runEvaluationsForKnnMethods(obj,xtrain,ytrain,xtest,ytest,model_dir)
      % ----------------------------------------------------------------------------------
      % Avalia os modelos dos métodos knn, lmnn e klmnn
      % ----------------------------------------------------------------------------------
      % Testes
      for K = 1:5
        for kappa = K:-1:K-3
          if kappa >= 1
            fprintf('\nK = %d \tkappa = %d\n',K,kappa);
            Manager.runEvaluations(obj,xtrain,ytrain,xtest,ytest,obj.method,...
              model_dir,num_experiments,K,kappa);
          end
        end
      end
    end
    
    function runEvaluationsParameter(obj,xtrain,ytrain,xtest,ytest,model_dir)
      % ----------------------------------------------------------------------------------
      % Avalia os métodos em um conjunto de testes
      % qualquer passando manualmente os parâmetros
      % ----------------------------------------------------------------------------------    
      knn_dir = strcat(model_dir,'/K=',int2str(obj.knn_arg),' kappa=',int2str(obj.knn_threshold));
      if ~exist(knn_dir,'dir')
        mkdir(knn_dir);
      end
      for i=1:numel(obj.method)
        switch obj.method{i}
          case 'knn'
            fprintf('\n-> KNN Novelty Detection\n\n');
            % Avalia os parâmetros
            t0_knn = tic;
            knn = KnnND(obj,xtrain,ytrain,knn_arg,knn_threshold,obj.num_classes);
            knn_evaluation = knn.evaluate(obj,xtrain,ytrain,xtest,ytest,...
              obj.parameters{1}.threshold_arg);
            knn_evaluation.evaluation_time = toc(t0_knn);
            % Salva avaliação
            save(strcat(knn_dir,'/knn_evaluate_parameter.mat'),'-struct','knn_evaluation');
          case 'lmnn'
            fprintf('\n-> LMNN Novelty Detection\n');
            % Avalia os parâmetros
            t0_lmnn = tic;
            lmnn = LmnnND(obj,xtrain,ytrain,knn_arg,knn_threshold,obj.num_classes);
            lmnn_evaluation = lmnn.evaluate(obj,xtrain,ytrain,xtest,ytest,...
              obj.parameters{2}.threshold_arg);
            lmnn_evaluation.evaluation_time = toc(t0_lmnn);
            % Salva avaliação
            save(strcat(knn_dir,'/lmnn_evaluate_parameter.mat'),...
              '-struct','lmnn_evaluation');
          case 'klmnn'
            fprintf('\n-> KLMNN Novelty Detection\n');
            % Avalia os parâmetros
            t0_klmnn = tic;
            klmnn = KlmnnND(obj,xtrain,ytrain,knn_arg,knn_threshold,obj.num_classes);
            klmnn.kernel_type = obj.parameters{3}.kernel_type;
            klmnn_evaluation = klmnn.evaluate(obj,xtrain,ytrain,xtest,ytest,...
              obj.parameters{3}.kernel_arg,obj.parameters{3}.threshold_arg);
            klmnn_evaluation.evaluation_time = toc(t0_klmnn);
            % Salva avaliação
            save(strcat(knn_dir,'/klmnn_evaluate_parameter.mat'),...
              '-struct','klmnn_evaluation');
          case 'knfst'
            fprintf('\n-> KNFST Novelty Detection\n');
            % Avalia os parâmetros
            t0_knfst = tic;
            knfst = KnfstND(obj,xtrain,ytrain,obj.num_classes);
            knfst.kernel_type = obj.parameters{4}.kernel_type;
            knfst_evaluation = knfst.evaluate(obj,xtrain,ytrain,xtest,ytest,...
              obj.parameters{4}.kernel_arg,obj.parameters{4}.threshold_arg);
            knfst_evaluation.evaluation_time = toc(t0_knfst);
            % Salva avaliação
            save(strcat(model_dir,'/knfst_evaluate_parameter.mat'),...
              '-struct','knfst_evaluation');
          case 'one_svm'
            fprintf('\n-> One SVM Novelty Detection\n');
            % Avalia os parâmetros
            t0_svm = tic;
            one_svm = SvmND(obj,xtrain,ytrain,obj.num_classes);
            one_svm.kernel_type = obj.parameters{5}.kernel_type;
            one_svm_evaluation = one_svm.evaluateOneClassSVM(obj,xtrain,ytrain,...
              xtest,ytest,obj.parameters{5}.kernel_arg);
            one_svm_evaluation.evaluation_time = toc(t0_svm);
            % Salva avaliação
            save(strcat(model_dir,'/one_svm_evaluate_parameter.mat'),...
              '-struct','one_svm_evaluation');
          case 'multi_svm'
            fprintf('\n-> Multi SVM Novelty Detection\n');
            % Avalia os parâmetros
            t0_svm = tic;
            multi_svm = SvmND(obj,xtrain,ytrain,obj.num_classes);
            multi_svm.kernel_type = obj.parameters{6}.kernel_type;
            multi_svm_evaluation = multi_svm.evaluateMultiClassSVM(obj,xtrain,ytrain,...
              xtest,ytest,obj.parameters{6}.kernel_arg,obj.parameters{6}.threshold_arg);
            multi_svm_evaluation.evaluation_time = toc(t0_svm);
            % Salva avaliação
            save(strcat(model_dir,'/multi_svm_evaluate_parameter.mat'),...
              '-struct','multi_svm_evaluation');
          case 'kpca'
            fprintf('\n-> KPCA Novelty Detection\n');
            % Avalia os parâmetros
            t0_kpca = tic;
            kpca = KpcaND(obj,xtrain,ytrain,obj.num_classes);
            kpca.kernel_type = obj.parameters{7}.kernel_type;
            kpca_evaluation = kpca.evaluate(obj,xtrain,xtest,ytest,...
              obj.parameters{7}.kernel_arg,obj.parameters{7}.threshold_arg);
            kpca_evaluation.evaluation_time = toc(t0_kpca);
            % Salva avaliação
            save(strcat(model_dir,'/kpca_evaluate_parameter.mat'),...
              '-struct','kpca_evaluation');
        end
      end
    end
    
    function runPredictions(obj,xtest,model_dir)
      % ----------------------------------------------------------------------------------
      % Executa as predições dos métodos em um conjunto de testes
      % qualquer com os parâmetros ótimos
      % ----------------------------------------------------------------------------------      
      xtrain = obj.X(obj.y~=-1,:);
      ytrain = obj.y(obj.y~=-1);
      knn_dir = strcat(model_dir,'/K=',int2str(obj.knn_arg),' kappa=',int2str(obj.knn_threshold));
      if ~exist(knn_dir,'dir')
        mkdir(knn_dir);
      end
      for i=1:numel(obj.method)
        switch obj.method{i}
          case 'knn'
            fprintf('\n-> KNN Novelty Detection \n');
            % Carrega o modelo
            knn_model = load(strcat(knn_dir,'/knn_model.mat'));
            knn_model = knn_model.knn_model;
            % Avalia o modelo
            t0_knn = tic;
            knn = KnnND(obj,xtrain,ytrain,knn_model.obj.knn_arg,knn_model.obj.knn_threshold,obj.num_classes);
            predictions = knn.predict(obj,xtrain,ytrain,xtest,knn_model.threshold);
            prediction_time = toc(t0_knn);
            fprintf('\n--> Ok [%.4f s]\n',prediction_time);
            % Salva a predição
            save(strcat(knn_dir,'/knn_predictions.mat'),...
              'prediction_time','predictions','xtest');
            % Plota a fronteira de decisão
            figure;
            Util.plotDecisionBoundary(xtest,predictions);
            Util.plotClassesAux(obj.X,obj.y);
          case 'lmnn'
            fprintf('\n-> LMNN Novelty Detection\n');
            % Carrega o modelo
            lmnn_model = load(strcat(knn_dir,'/lmnn_model.mat'));
            lmnn_model = lmnn_model.lmnn_model;
            % Avalia o modelo
            t0_lmnn = tic;
            lmnn = LmnnND(obj,xtrain,ytrain,lmnn_model.obj.knn_arg,...
              lmnn_model.obj.knn_threshold,obj.num_classes);
            predictions = lmnn.predict(obj,xtrain,ytrain,xtest,lmnn_model.threshold);
            prediction_time = toc(t0_lmnn);
            fprintf('\n--> Ok [%.4f s]\n',prediction_time);
            % Salva o teste
            save(strcat(knn_dir,'/lmnn_predictions.mat'),...
              'prediction_time','predictions','xtest');
            % Plota a fronteira de decisão
            figure;
            Util.plotDecisionBoundary(xtest,predictions);
            Util.plotClassesAux(obj.X,obj.y);
          case 'klmnn'
            fprintf('\n-> KLMNN Novelty Detection\n');
            % Carrega o modelo
            klmnn_model = load(strcat(knn_dir,'/klmnn_model.mat'));
            klmnn_model = klmnn_model.klmnn_model;
            % Avalia o modelo
            t0_klmnn = tic;
            klmnn = KlmnnND(obj,xtrain,ytrain,klmnn_model.obj.knn_arg,...
              klmnn_model.obj.knn_threshold,obj.num_classes);
            klmnn.kernel_type = klmnn_model.kernel_type;
            %klmnn_predictions = klmnn.predict(obj,xtrain,ytrain,xtest,...
            % klmnn_model.kernel,klmnn_model.threshold);
            predictions = klmnn.predict(obj,xtrain,ytrain,xtest,...
              klmnn_model.kernel,klmnn_model.threshold);
            prediction_time = toc(t0_klmnn);
            fprintf('-> done! [%.4f s]\n',prediction_time);
            % Salva o teste
            save(strcat(knn_dir,'/klmnn_predictions.mat'),...
              'prediction_time','predictions','xtest');
            % Plota a fronteira de decisão
            figure;
            Util.plotDecisionBoundary(xtest,predictions);
            Util.plotClassesAux(obj.X,obj.y);
          case 'knfst'
            fprintf('\n-> KNFST Novelty Detection\n');
            % Carrega o modelo
            knfst_model = load(strcat(model_dir,'/knfst_model.mat'));
            knfst_model = knfst_model.knfst_model;
            % Avalia o modelo
            t0_knfst = tic;
            knfst = KnfstND(obj,xtrain,ytrain,obj.num_classes);
            knfst.kernel_type = knfst_model.kernel_type;
            predictions = knfst.predict(obj,xtrain,ytrain,xtest,...
              knfst_model.kernel,knfst_model.threshold);
            prediction_time = toc(t0_knfst);
            fprintf('\n--> Ok [%.4f s]\n',prediction_time);
            % Salva o teste
            save(strcat(model_dir,'/knfst_predictions.mat'),...
              'prediction_time','predictions','xtest');
            % Plota a fronteira de decisão
            figure;
            Util.plotDecisionBoundary(xtest,predictions);
            Util.plotClassesAux(obj.X,obj.y);
          case 'one_svm'
            fprintf('\n-> One SVM Novelty Detection\n');
            % Carrega o modelo
            one_svm_model = load(strcat(model_dir,'/one_svm_model.mat'));
            one_svm_model = one_svm_model.one_svm_model;
            % Avalia o modelo
            t0_svm = tic;
            one_svm = SvmND(obj,xtrain,ytrain,obj.num_classes);
            one_svm.kernel_type = one_svm_model.kernel_type;
            predictions = one_svm.predictOneClassSVM(obj,xtrain,ytrain,xtest,...
              one_svm_model.kernel);
            prediction_time = toc(t0_svm);
            fprintf('\n--> Ok [%.4f s]\n',prediction_time);
            % Salva o teste
            save(strcat(model_dir,'/one_svm_predictions.mat'),'prediction_time',...
              'predictions','xtest');
            % Plota a fronteira de decisão
            figure;
            Util.plotDecisionBoundary(xtest,predictions);
            Util.plotClassesAux(obj.X,obj.y);
          case 'multi_svm'
            fprintf('\n-> Multi SVM Novelty Detection\n');
            % Carrega o modelo
            multi_svm_model = load(strcat(model_dir,'/multi_svm_model.mat'));
            multi_svm_model = multi_svm_model.multi_svm_model;
            % Avalia o modelo
            t0_svm = tic;
            multi_svm = SvmND(obj,xtrain,ytrain,obj.num_classes);
            multi_svm.kernel_type = multi_svm_model.kernel_type;
            predictions = multi_svm.predictMultiClassSVM(obj,xtrain,ytrain,xtest,...
              multi_svm_model.kernel,multi_svm_model.threshold);
            prediction_time = toc(t0_svm);
            fprintf('\n--> Ok [%.4f s]\n',prediction_time);
            % Salva o teste
            save(strcat(model_dir,'/multi_svm_predictions.mat'),'prediction_time',...
              'predictions','xtest');
            % Plota a fronteira de decisão
            figure;
            Util.plotDecisionBoundary(xtest,predictions);
            Util.plotClassesAux(obj.X,obj.y);
          case 'kpca'
            fprintf('\n-> KPCA Novelty Detection\n');
            % Carrega o modelo
            kpca_model = load(strcat(model_dir,'/kpca_model.mat'));
            kpca_model = kpca_model.kpca_model;
            % Avalia o modelo
            t0_kpca = tic;
            kpca = KpcaND(obj,xtrain,ytrain,obj.num_classes);
            kpca.kernel_type = kpca_model.kernel_type;
            predictions = kpca.predictNovelty(obj,xtrain,xtest,...
              kpca_model.kernel,kpca_model.threshold);
            prediction_time = toc(t0_kpca);
            fprintf('\n--> Ok [%.4f s]\n',prediction_time);
            % Salva o teste
            save(strcat(model_dir,'/kpca_predictions.mat'),...
              'prediction_time','predictions','xtest');
            % Plota a fronteira de decisão
            figure;
            Util.plotDecisionBoundary(xtest,predictions);
            Util.plotClassesAux(obj.X,obj.y);
        end
      end
    end
    
    function runPredictionsForKnnMethods(obj,xtrain,ytrain,xtest,ytest,model_dir)
      % ----------------------------------------------------------------------------------
      % Avalia os modelos dos métodos knn, lmnn e klmnn usando uma parte das amostras não
      % utilizadas na validação
      % ----------------------------------------------------------------------------------    
      % Testes
      for K = 1:5
        for kappa = K:-1:K-3
          if kappa >= 1
            fprintf('\nK = %d \tkappa = %d\n',K,kappa);
            Manager.runPredict(obj,xtrain,ytrain,xtest,ytest,obj.method,...
              model_dir,num_experiments,K,kappa);
          end
        end
      end
    end
    
    function runPredictionsParameter(obj,xtest,model_dir)
      % ----------------------------------------------------------------------------------
      % Executa as predições dos métodos em um conjunto de testes
      % qualquer passando manualmente os parâmetros
      % ----------------------------------------------------------------------------------    
      xtrain = obj.X(obj.y~=-1,:);
      ytrain = obj.y(obj.y~=-1);
      knn_dir = strcat(model_dir,'/K=',int2str(obj.knn_arg),' kappa=',int2str(obj.knn_threshold));
      if ~exist(knn_dir,'dir')
        mkdir(knn_dir);
      end
      for i=1:numel(obj.method)
        switch obj.method{i}
          case 'knn'
            fprintf('\n-> KNN Novelty Detection \n');
            t0_knn = tic;
            knn = KnnND(obj,xtrain,ytrain,knn_arg,knn_threshold,obj.num_classes);
            % Avalia os parâmetros
            knn_predictions = knn.predict(obj,xtrain,ytrain,xtest,obj.parameters{1}.threshold_arg);
            prediction_time = toc(t0_knn);
            fprintf('\n--> Ok [%.4f s]\n',prediction_time);
            % Plota a fronteira de decisão
            Util.plotClassesWithBoundary('knn',knn_dir,obj.X,obj.y,xtest,knn_predictions);
          case 'lmnn'
            fprintf('\n-> LMNN Novelty Detection\n');
            t0_lmnn = tic;
            lmnn = LmnnND(obj,xtrain,ytrain,knn_arg,knn_threshold,obj.num_classes);
            % Avalia os parâmetros
            lmnn_predictions = lmnn.predict(obj,xtrain,ytrain,xtest,obj.parameters{2}.threshold_arg);
            prediction_time = toc(t0_lmnn);
            fprintf('\n--> Ok [%.4f s]\n',prediction_time);
            % Plota a fronteira de decisão
            Util.plotClassesWithBoundary('lmnn',knn_dir,obj.X,obj.y,xtest,lmnn_predictions);
          case 'klmnn'
            fprintf('\n-> KLMNN Novelty Detection\n');
            t0_klmnn = tic;
            klmnn = KlmnnND(obj,xtrain,ytrain,knn_arg,knn_threshold,obj.num_classes);
            klmnn.kernel_type = obj.parameters{3}.kernel_type;
            % Avalia os parâmetros
            klmnn_predictions = klmnn.predict(obj,xtrain,ytrain,xtest,...
              obj.parameters{3}.kernel_arg,obj.parameters{3}.threshold_arg);
            prediction_time = toc(t0_klmnn);
            fprintf('-> done! [%.4f s]\n',prediction_time);
            % Plota a fronteira de decisão
            Util.plotClassesWithBoundary('klmnn',knn_dir,obj.X,obj.y,xtest,klmnn_predictions);
          case 'knfst'
            fprintf('\n-> KNFST Novelty Detection\n');
            t0_knfst = tic;
            knfst = KnfstND(obj,xtrain,ytrain,obj.num_classes);
            knfst.kernel_type = obj.parameters{4}.kernel_type;
            % Avalia os parâmetros
            knfst_predictions = knfst.predict(obj,xtrain,ytrain,xtest,...
              obj.parameters{4}.kernel_arg,obj.parameters{4}.threshold_arg);
            prediction_time = toc(t0_knfst);
            fprintf('\n--> Ok [%.4f s]\n',prediction_time);
            % Plota a fronteira de decisão
            Util.plotClassesWithBoundary('knfst',model_dir,obj.X,obj.y,xtest,knfst_predictions);
          case 'one_svm'
            fprintf('\n-> One SVM Novelty Detection\n');
            t0_svm = tic;
            one_svm = SvmND(obj,xtrain,ytrain,obj.num_classes);
            multi_svm.kernel_type = obj.parameters{5}.kernel_type;
            % Testa o parâmetro
            one_svm_predictions = one_svm.predictOneClassSVM(obj,xtrain,ytrain,xtest,...
              obj.parameters{5}.kernel_arg);
            prediction_time = toc(t0_svm);
            fprintf('\n--> Ok [%.4f s]\n',prediction_time);
            % Plota a fronteira de decisão
            Util.plotClassesWithBoundary('one svm',model_dir,obj.X,obj.y,xtest,one_svm_predictions);
          case 'multi_svm'
            fprintf('\n-> Multi SVM Novelty Detection\n');
            t0_svm = tic;
            multi_svm = SvmND(obj,xtrain,ytrain,obj.num_classes);
            multi_svm.kernel_type = obj.parameters{6}.kernel_type;
            % Avalia os parâmetros
            multi_svm_predictions = multi_svm.predictMultiClassSVM(obj,xtrain,ytrain,xtest,...
              obj.parameters{6}.kernel_arg,obj.parameters{6}.threshold_arg);
            prediction_time = toc(t0_svm);
            fprintf('\n--> Ok [%.4f s]\n',prediction_time);
            % Plota a fronteira de decisão
            Util.plotClassesWithBoundary('multi svm',model_dir,obj.X,obj.y,xtest,multi_svm_predictions);
          case 'kpca'
            fprintf('\n-> KPCA Novelty Detection\n');
            t0_kpca = tic;
            kpca = KpcaND(obj,xtrain,ytrain,obj.num_classes);
            kpca.kernel_type = obj.parameters{7}.kernel_type;
            % Avalia os parâmetros
            kpca_predictions = kpca.predictNovelty(obj,xtrain,xtest,...
              obj.parameters{7}.kernel_arg,obj.parameters{7}.threshold_arg);
            prediction_time = toc(t0_kpca);
            fprintf('\n--> Ok [%.4f s]\n',prediction_time);
            % Plota a fronteira de decisão
            Util.plotClassesWithBoundary('kpca',model_dir,obj.X,obj.y,xtest,kpca_predictions);
        end
      end
    end
    
    function runPredictionsParametersForKnnMethods(obj,xtrain,ytrain,xtest,ytest,model_dir)
      % ----------------------------------------------------------------------------------
      % Avalia os modelos dos métodos knn, lmnn e klmnn usando uma parte das amostras não
      % utilizadas na validação
      % ----------------------------------------------------------------------------------    
      % Testes
      for K = 1:5
        for kappa = K:-1:K-3
          if kappa >= 1
            fprintf('\nK = %d \tkappa = %d\n',K,kappa);
            Manager.runPredictParameter(obj,xtrain,ytrain,xtest,ytest,obj.method,...
              model_dir,obj.num_experiments,K,kappa);
          end
        end
      end
    end

    function reportExperimentResults(obj,model_dir)
      % ----------------------------------------------------------------------------------
      % Carrega e processa os resultados dos experimentos de detecção de
      % novidade
      % ----------------------------------------------------------------------------------      
      fprintf('\nProcessing results... ');
      knn_dir = strcat(model_dir,'/K=',int2str(obj.knn_arg),' kappa=',int2str(obj.knn_threshold));
      TPR = zeros(1,7);
      TNR = zeros(1,7);
      FPR = zeros(1,7);
      FNR = zeros(1,7);
      AFR = zeros(1,7);
      F1 = zeros(1,7);
      MCC = zeros(1,7);
      for k=1:numel(obj.method)
        switch obj.method{k}
          case 'knn'
            try
              experiment = load(strcat(knn_dir,'/knn_experiments.mat'));
              TPR(1) = experiment.tpr_score;
              TNR(1) = experiment.tnr_score;
              FPR(1) = experiment.fpr_score;
              FNR(1) = experiment.fnr_score;
              AFR(1) = experiment.afr_score;
              F1(1) = experiment.f1_score;
              MCC(1) = experiment.mcc_score;
            catch
              fprintf('\n--> error processing knn results\n');
            end
          case 'lmnn'
            try
              experiment = load(strcat(knn_dir,'/lmnn_experiments.mat'));
              TPR(2) = experiment.tpr_score;
              TNR(2) = experiment.tnr_score;
              FPR(2) = experiment.fpr_score;
              FNR(2) = experiment.fnr_score;
              AFR(2) = experiment.afr_score;
              F1(2) = experiment.f1_score;
              MCC(2) = experiment.mcc_score;
            catch
              fprintf('\n--> error processing lmnn results\n');
            end
          case 'klmnn'
            try
              experiment = load(strcat(knn_dir,'/klmnn_experiments.mat'));
              TPR(3) = experiment.tpr_score;
              TNR(3) = experiment.tnr_score;
              FPR(3) = experiment.fpr_score;
              FNR(3) = experiment.fnr_score;
              AFR(3) = experiment.afr_score;
              F1(3) = experiment.f1_score;
              MCC(3) = experiment.mcc_score;
            catch
              fprintf('\n--> error processing klmnn results\n');
            end
          case 'knfst'
            try
              experiment = load(strcat(model_dir,'/knfst_experiments.mat'));
              TPR(4) = experiment.tpr_score;
              TNR(4) = experiment.tnr_score;
              FPR(4) = experiment.fpr_score;
              FNR(4) = experiment.fnr_score;
              AFR(4) = experiment.afr_score;
              F1(4) = experiment.f1_score;
              MCC(4) = experiment.mcc_score;
            catch
              fprintf('\n--> error processing knfst results\n');
            end
          case 'one_svm'
            try
              experiment = load(strcat(model_dir,'/one_svm_experiments.mat'));
              TPR(5) = experiment.tpr_score;
              TNR(5) = experiment.tnr_score;
              FPR(5) = experiment.fpr_score;
              FNR(5) = experiment.fnr_score;
              AFR(5) = experiment.afr_score;
              F1(5) = experiment.f1_score;
              MCC(5) = experiment.mcc_score;
            catch
              fprintf('\n--> error processing one svm results\n');
            end
          case 'multi_svm'
            try
              experiment = load(strcat(model_dir,'/multi_svm_experiments.mat'));
              TPR(6) = experiment.tpr_score;
              TNR(6) = experiment.tnr_score;
              FPR(6) = experiment.fpr_score;
              FNR(6) = experiment.fnr_score;
              AFR(6) = experiment.afr_score;
              F1(6) = experiment.f1_score;
              MCC(6) = experiment.mcc_score;
            catch
              fprintf('\n--> erro processing multi svm results\n');
            end
          case 'kpca'
            try
              experiment = load(strcat(model_dir,'/kpca_experiments.mat'));
              TPR(7) = experiment.tpr_score;
              TNR(7) = experiment.tnr_score;
              FPR(7) = experiment.fpr_score;
              FNR(7) = experiment.fnr_score;
              AFR(7) = experiment.afr_score;
              F1(7) = experiment.f1_score;
              MCC(7) = experiment.mcc_score;
            catch
              fprintf('\n--> error processing kpca results\n');
            end
        end
      end
      
      method_names = {'KNN','LMNN','KLMNN','KNFST','ONE_SVM','MULTI_SVM','KPCA_NOV'};
      metric_names = {'TPR','TNR','FPR','FNR','AFR','F1','MCC'};
      
      REPORT = [TPR; TNR; FPR; FNR; AFR; F1; MCC];
      
      REPORT = array2table(round(REPORT,2));
      REPORT.Properties.VariableNames = method_names;
      REPORT.Properties.RowNames = metric_names;
      
      save(strcat(model_dir,'/report_results.mat'),'TPR','TNR',...
        'FPR','FNR','AFR','F1','MCC','REPORT');
      writetable(REPORT,strcat(model_dir,'/report_results.csv'),...
        'WriteRowNames',true,'Delimiter',';');
      fprintf('done!\n');
    end
    
    function reportExperimentsForKnnMethods(obj,model_dir)
      % ----------------------------------------------------------------------------------
      % Carrega e processa as métricas de acurácia dos métodos knn,
      % lmnn e klmnn para vários parâmetros de K e kappa
      % ----------------------------------------------------------------------------------      
      fprintf('\nLoading experiment results... ');
      % KNN
      KNN.TPR = nan*zeros(5,4); KNN.TNR = nan*zeros(5,4);
      KNN.FPR = nan*zeros(5,4); KNN.FNR = nan*zeros(5,4);
      KNN.AFR = nan*zeros(5,4); KNN.F1 = nan*zeros(5,4);
      KNN.MCC = nan*zeros(5,4);
      % LMNN
      LMNN.TPR = nan*zeros(5,4); LMNN.TNR = nan*zeros(5,4);
      LMNN.FPR = nan*zeros(5,4); LMNN.FNR = nan*zeros(5,4);
      LMNN.AFR = nan*zeros(5,4); LMNN.F1 = nan*zeros(5,4);
      LMNN.MCC = nan*zeros(5,4);
      % KLMNN
      KLMNN.TPR = nan*zeros(5,4); KLMNN.TNR = nan*zeros(5,4);
      KLMNN.FPR = nan*zeros(5,4); KLMNN.FNR = nan*zeros(5,4);
      KLMNN.AFR = nan*zeros(5,4); KLMNN.F1 = nan*zeros(5,4);
      KLMNN.MCC = nan*zeros(5,4);
      
      % Testes
      for K = 1:5
        for kappa = K:-1:K-3
          if kappa >= 1
            j = K-kappa+1;
            knn_dir = strcat(model_dir,'/K=',int2str(K),' kappa=',int2str(kappa));
            fprintf('\nK = %d \tkappa = %d\n',K,kappa);
            % KNN
            try
              file = strcat(knn_dir,'/knn_experiments.mat');
              experiment = load(file);
              KNN.TPR(K,j) = experiment.tpr_score;
              KNN.TNR(K,j) = experiment.tnr_score;
              KNN.FPR(K,j) = experiment.fpr_score;
              KNN.FNR(K,j) = experiment.fnr_score;
              KNN.AFR(K,j) = experiment.afr_score;
              KNN.F1(K,j) = experiment.f1_score;
              KNN.MCC(K,j) = experiment.mcc_score;
            catch
              fprintf('\n--> error knn results!\n');
            end
            
            % LMNN
            try
              file = strcat(knn_dir,'/lmnn_experiments.mat');
              experiment = load(file);
              LMNN.TPR(K,j) = experiment.tpr_score;
              LMNN.TNR(K,j) = experiment.tnr_score;
              LMNN.FPR(K,j) = experiment.fpr_score;
              LMNN.FNR(K,j) = experiment.fnr_score;
              LMNN.AFR(K,j) = experiment.afr_score;
              LMNN.F1(K,j) = experiment.f1_score;
              LMNN.MCC(K,j) = experiment.mcc_score;
            catch
              fprintf('\n--> error lmnn results!\n');
            end
            
            % KLMNN
            try
              file = strcat(knn_dir,'/klmnn_experiments.mat');
              experiment = load(file);
              KLMNN.TPR(K,j) = experiment.tpr_score;
              KLMNN.TNR(K,j) = experiment.tnr_score;
              KLMNN.FPR(K,j) = experiment.fpr_score;
              KLMNN.FNR(K,j) = experiment.fnr_score;
              KLMNN.AFR(K,j) = experiment.afr_score;
              KLMNN.F1(K,j) = experiment.f1_score;
              KLMNN.MCC(K,j) = experiment.mcc_score;
            catch
              fprintf('\n--> error klmnn results!\n');
            end
          end
        end
      end
      
      % Cria as tabelas
      K_names = split(sprintf('K = %d,',1:5),',');
      K_names = K_names(1:end-1)';
      kappa_names = {'kappa1','kappa2','kappa3','kappa4'};
      
      kappa = struct('kappa1','K','kappa2','K-1','kappa3','K-2','kappa4','K-3');
      
      % KNN
      KNN.TPR = array2table(round(KNN.TPR,4),'VariableNames',kappa_names,'RowNames',K_names);
      KNN.TNR = array2table(round(KNN.TNR,4),'VariableNames',kappa_names,'RowNames',K_names);
      KNN.FPR = array2table(round(KNN.FPR,4),'VariableNames',kappa_names,'RowNames',K_names);
      KNN.FNR = array2table(round(KNN.FNR,4),'VariableNames',kappa_names,'RowNames',K_names);
      KNN.AFR = array2table(round(KNN.AFR,4),'VariableNames',kappa_names,'RowNames',K_names);
      KNN.F1 = array2table(round(KNN.F1,4),'VariableNames',kappa_names,'RowNames',K_names);
      KNN.MCC = array2table(round(KNN.MCC,4),'VariableNames',kappa_names,'RowNames',K_names);
      
      % LMNN
      LMNN.TPR = array2table(round(LMNN.TPR,4),'VariableNames',kappa_names,'RowNames',K_names);
      LMNN.TNR = array2table(round(LMNN.TNR,4),'VariableNames',kappa_names,'RowNames',K_names);
      LMNN.FPR = array2table(round(LMNN.FPR,4),'VariableNames',kappa_names,'RowNames',K_names);
      LMNN.FNR = array2table(round(LMNN.FNR,4),'VariableNames',kappa_names,'RowNames',K_names);
      LMNN.AFR = array2table(round(LMNN.AFR,4),'VariableNames',kappa_names,'RowNames',K_names);
      LMNN.F1 = array2table(round(LMNN.F1,4),'VariableNames',kappa_names,'RowNames',K_names);
      LMNN.MCC = array2table(round(LMNN.MCC,4),'VariableNames',kappa_names,'RowNames',K_names);
      
      % KLMNN
      KLMNN.TPR = array2table(round(KLMNN.TPR,4),'VariableNames',kappa_names,'RowNames',K_names);
      KLMNN.TNR = array2table(round(KLMNN.TNR,4),'VariableNames',kappa_names,'RowNames',K_names);
      KLMNN.FPR = array2table(round(KLMNN.FPR,4),'VariableNames',kappa_names,'RowNames',K_names);
      KLMNN.FNR = array2table(round(KLMNN.FNR,4),'VariableNames',kappa_names,'RowNames',K_names);
      KLMNN.AFR = array2table(round(KLMNN.AFR,4),'VariableNames',kappa_names,'RowNames',K_names);
      KLMNN.F1 = array2table(round(KLMNN.F1,4),'VariableNames',kappa_names,'RowNames',K_names);
      KLMNN.MCC = array2table(round(KLMNN.MCC,4),'VariableNames',kappa_names,'RowNames',K_names);
      
      save(strcat(model_dir,'/report_knn_obj.method.mat'),'kappa','KNN','LMNN','KLMNN');
      fprintf('done!\n');
    end
    
    function reportEvaluation(obj,model_dir)
      % ----------------------------------------------------------------------------------
      % Carrega e processa avaliações dos métodos de detecção de novidade
      % ----------------------------------------------------------------------------------      
      fprintf('\nProcessing results... ');
      knn_dir = strcat(model_dir,'/K=',int2str(obj.knn_arg),' kappa=',int2str(obj.knn_threshold));
      TPR = zeros(1,7);
      TNR = zeros(1,7);
      FPR = zeros(1,7);
      FNR = zeros(1,7);
      AFR = zeros(1,7);
      F1 = zeros(1,7);
      MCC = zeros(1,7);
      for k=1:numel(obj.method)
        switch obj.method{k}
          case 'knn'
            try
              experiment = load(strcat(knn_dir,'/knn_evaluation.mat'));
              TPR(1) = experiment.TPR;
              TNR(1) = experiment.TNR;
              FPR(1) = experiment.FPR;
              FNR(1) = experiment.FNR;
              AFR(1) = experiment.AFR;
              F1(1) = experiment.F1;
              MCC(1) = experiment.MCC;
            catch
              fprintf('\n--> error processing knn results\n');
            end
          case 'lmnn'
            try
              experiment = load(strcat(knn_dir,'/lmnn_evaluation.mat'));
              TPR(2) = experiment.TPR;
              TNR(2) = experiment.TNR;
              FPR(2) = experiment.FPR;
              FNR(2) = experiment.FNR;
              AFR(2) = experiment.AFR;
              F1(2) = experiment.F1;
              MCC(2) = experiment.MCC;
            catch
              fprintf('\n--> error processing lmnn results\n');
            end
          case 'klmnn'
            try
              experiment = load(strcat(knn_dir,'/klmnn_evaluation.mat'));
              TPR(3) = experiment.TPR;
              TNR(3) = experiment.TNR;
              FPR(3) = experiment.FPR;
              FNR(3) = experiment.FNR;
              AFR(3) = experiment.AFR;
              F1(3) = experiment.F1;
              MCC(3) = experiment.MCC;
            catch
              fprintf('\n--> error processing klmnn results\n');
            end
          case 'knfst'
            try
              experiment = load(strcat(model_dir,'/knfst_evaluation.mat'));
              TPR(4) = experiment.TPR;
              TNR(4) = experiment.TNR;
              FPR(4) = experiment.FPR;
              FNR(4) = experiment.FNR;
              AFR(4) = experiment.AFR;
              F1(4) = experiment.F1;
              MCC(4) = experiment.MCC;
            catch
              fprintf('\n--> error processing knfst results\n');
            end
          case 'one_svm'
            try
              experiment = load(strcat(model_dir,'/one_svm_evaluation.mat'));
              TPR(5) = experiment.TPR;
              TNR(5) = experiment.TNR;
              FPR(5) = experiment.FPR;
              FNR(5) = experiment.FNR;
              AFR(5) = experiment.AFR;
              F1(5) = experiment.F1;
              MCC(5) = experiment.MCC;
            catch
              fprintf('\n--> error processing one svm results\n');
            end
          case 'multi_svm'
            try
              experiment = load(strcat(model_dir,'/multi_svm_evaluation.mat'));
              TPR(6) = experiment.TPR;
              TNR(6) = experiment.TNR;
              FPR(6) = experiment.FPR;
              FNR(6) = experiment.FNR;
              AFR(6) = experiment.AFR;
              F1(6) = experiment.F1;
              MCC(6) = experiment.MCC;
            catch
              fprintf('\n--> erro processing multi svm results\n');
            end
          case 'kpca'
            try
              experiment = load(strcat(model_dir,'/kpca_evaluation.mat'));
              TPR(7) = experiment.TPR;
              TNR(7) = experiment.TNR;
              FPR(7) = experiment.FPR;
              FNR(7) = experiment.FNR;
              AFR(7) = experiment.AFR;
              F1(7) = experiment.F1;
              MCC(7) = experiment.MCC;
            catch
              fprintf('\n--> error processing kpca results\n');
            end
        end
      end
      
      method_names = {'KNN','LMNN','KLMNN','KNFST','ONE_SVM','MULTI_SVM','KPCA_NOV'};
      metric_names = {'TPR','TNR','FPR','FNR','AFR','F1','MCC'};
      
      REPORT = [TPR; TNR; FPR; FNR; AFR; F1; MCC];
      
      REPORT = array2table(round(REPORT,2));
      REPORT.Properties.VariableNames = method_names;
      REPORT.Properties.RowNames = metric_names;
      
      save(strcat(model_dir,'/report_results.mat'),'TPR','TNR',...
        'FPR','FNR','AFR','F1','MCC','REPORT');
      writetable(REPORT,strcat(model_dir,'/report_results.csv'),...
        'WriteRowNames',true,'Delimiter',';');
      fprintf('done!\n');
    end
    
    function reportEvaluations(obj,model_dir)
      % ----------------------------------------------------------------------------------
      % Carrega e processa avaliações dos métodos de detecção de novidade
      % ----------------------------------------------------------------------------------      
      fprintf('\nProcessing results... ');
      knn_dir = strcat(model_dir,'/K=',int2str(obj.knn_arg),' kappa=',int2str(obj.knn_threshold));
      TPR = zeros(obj.num_experiments,7);
      TNR = zeros(obj.num_experiments,7);
      FPR = zeros(obj.num_experiments,7);
      FNR = zeros(obj.num_experiments,7);
      AFR = zeros(obj.num_experiments,7);
      ACC = zeros(obj.num_experiments,7);
      F1 = zeros(obj.num_experiments,7);
      MCC = zeros(obj.num_experiments,7);
      for j=1:obj.num_experiments
        for k=1:numel(obj.method)
          switch obj.method{k}
            case 'knn'
              try
                knn_evaluations = load(strcat(knn_dir,'/knn_evaluations.mat'));
                knn_evaluations = knn_evaluations.knn_evaluations;
                TPR(j,1) = knn_evaluations.evaluations{j}.TPR;
                TNR(j,1) = knn_evaluations.evaluations{j}.TNR;
                FPR(j,1) = knn_evaluations.evaluations{j}.FPR;
                FNR(j,1) = knn_evaluations.evaluations{j}.FNR;
                AFR(j,1) = knn_evaluations.evaluations{j}.AFR;
                ACC(j,1) = knn_evaluations.evaluations{j}.ACC;
                F1(j,1) = knn_evaluations.evaluations{j}.F1;
                MCC(j,1) = knn_evaluations.evaluations{j}.MCC;
              catch
                fprintf('\n--> error processing knn results\n');
              end
            case 'lmnn'
              try
                lmnn_evaluations = load(strcat(knn_dir,'/lmnn_evaluations.mat'));
                lmnn_evaluations = lmnn_evaluations.lmnn_evaluations;
                TPR(j,2) = lmnn_evaluations.evaluations{j}.TPR;
                TNR(j,2) = lmnn_evaluations.evaluations{j}.TNR;
                FPR(j,2) = lmnn_evaluations.evaluations{j}.FPR;
                FNR(j,2) = lmnn_evaluations.evaluations{j}.FNR;
                AFR(j,2) = lmnn_evaluations.evaluations{j}.AFR;
                ACC(j,2) = lmnn_evaluations.evaluations{j}.ACC;
                F1(j,2) = lmnn_evaluations.evaluations{j}.F1;
                MCC(j,2) = lmnn_evaluations.evaluations{j}.MCC;
              catch
                fprintf('\n--> error processing lmnn results\n');
              end
            case 'klmnn'
              try
                klmnn_evaluations = load(strcat(model_dir,'/K=',int2str(obj.knn_arg),...
                  ' kappa=',int2str(obj.knn_threshold),'/klmnn_evaluations.mat'));
                klmnn_evaluations = klmnn_evaluations.klmnn_evaluations;
                TPR(j,3) = klmnn_evaluations.evaluations{j}.TPR;
                TNR(j,3) = klmnn_evaluations.evaluations{j}.TNR;
                FPR(j,3) = klmnn_evaluations.evaluations{j}.FPR;
                FNR(j,3) = klmnn_evaluations.evaluations{j}.FNR;
                AFR(j,3) = klmnn_evaluations.evaluations{j}.AFR;
                ACC(j,3) = klmnn_evaluations.evaluations{j}.ACC;
                F1(j,3) = klmnn_evaluations.evaluations{j}.F1;
                MCC(j,3) = klmnn_evaluations.evaluations{j}.MCC;
              catch
                fprintf('\n--> error processing klmnn results\n');
              end
            case 'knfst'
              try
                knfst_evaluations = load(strcat(model_dir,'/knfst_evaluations.mat'));
                knfst_evaluations = knfst_evaluations.knfst_evaluations;
                TPR(j,4) = knfst_evaluations.evaluations{j}.TPR;
                TNR(j,4) = knfst_evaluations.evaluations{j}.TNR;
                FPR(j,4) = knfst_evaluations.evaluations{j}.FPR;
                FNR(j,4) = knfst_evaluations.evaluations{j}.FNR;
                AFR(j,4) = knfst_evaluations.evaluations{j}.AFR;
                ACC(j,4) = knfst_evaluations.evaluations{j}.ACC;
                F1(j,4) = knfst_evaluations.evaluations{j}.F1;
                MCC(j,4) = knfst_evaluations.evaluations{j}.MCC;
              catch
                fprintf('\n--> error processing knfst results\n');
              end
            case 'one_svm'
              try
                one_svm_evaluations = load(strcat(model_dir,'/one_svm_evaluations.mat'));
                one_svm_evaluations = one_svm_evaluations.one_svm_evaluations;
                TPR(j,5) = one_svm_evaluations.evaluations{j}.TPR;
                TNR(j,5) = one_svm_evaluations.evaluations{j}.TNR;
                FPR(j,5) = one_svm_evaluations.evaluations{j}.FPR;
                FNR(j,5) = one_svm_evaluations.evaluations{j}.FNR;
                AFR(j,5) = one_svm_evaluations.evaluations{j}.AFR;
                ACC(j,5) = one_svm_evaluations.evaluations{j}.ACC;
                F1(j,5) = one_svm_evaluations.evaluations{j}.F1;
                MCC(j,5) = one_svm_evaluations.evaluations{j}.MCC;
              catch
                fprintf('\n--> error processing one svm results\n');
              end
            case 'multi_svm'
              try
                multi_svm_evaluations = load(strcat(model_dir,'/multi_svm_evaluations.mat'));
                multi_svm_evaluations = multi_svm_evaluations.multi_svm_evaluations;
                TPR(j,6) = multi_svm_evaluations.evaluations{j}.TPR;
                TNR(j,6) = multi_svm_evaluations.evaluations{j}.TNR;
                FPR(j,6) = multi_svm_evaluations.evaluations{j}.FPR;
                FNR(j,6) = multi_svm_evaluations.evaluations{j}.FNR;
                AFR(j,6) = multi_svm_evaluations.evaluations{j}.AFR;
                ACC(j,6) = multi_svm_evaluations.evaluations{j}.ACC;
                F1(j,6) = multi_svm_evaluations.evaluations{j}.F1;
                MCC(j,6) = multi_svm_evaluations.evaluations{j}.MCC;
              catch
                fprintf('\n--> erro processing multi svm results\n');
              end
            case 'kpca'
              try
                kpca_evaluations = load(strcat(model_dir,'/kpca_evaluations.mat'));
                kpca_evaluations = kpca_evaluations.kpca_evaluations;
                TPR(j,7) = kpca_evaluations.evaluations{j}.TPR;
                TNR(j,7) = kpca_evaluations.evaluations{j}.TNR;
                FPR(j,7) = kpca_evaluations.evaluations{j}.FPR;
                FNR(j,7) = kpca_evaluations.evaluations{j}.FNR;
                AFR(j,7) = kpca_evaluations.evaluations{j}.AFR;
                ACC(j,7) = kpca_evaluations.evaluations{j}.ACC;
                F1(j,7) = kpca_evaluations.evaluations{j}.F1;
                MCC(j,7) = kpca_evaluations.evaluations{j}.MCC;
              catch
                fprintf('\n--> error processing kpca results\n');
              end
          end
        end
      end
      
      test_names = split(sprintf('TEST%d,',1:obj.num_experiments),',');
      test_names = test_names(1:end-1);
      test_names{end+1} = 'MEAN';
      
      method_names = {'KNN','LMNN','KLMNN','KNFST','ONE_SVM','MULTI_SVM','KPCA_NOV'};
      
      % TPR
      TPR = array2table([TPR;mean(TPR,1)]);
      TPR.Properties.VariableNames = method_names;
      TPR.Properties.RowNames = test_names;
      
      % TNR
      TNR = array2table([TNR;mean(TNR,1)]);
      TNR.Properties.VariableNames = method_names;
      TNR.Properties.RowNames = test_names;
      
      % FPR
      FPR = array2table([FPR;mean(FPR,1)]);
      FPR.Properties.VariableNames = method_names;
      FPR.Properties.RowNames = test_names;
      
      % FNR
      FNR = array2table([FNR;mean(FNR,1)]);
      FNR.Properties.VariableNames = method_names;
      FNR.Properties.RowNames = test_names;
      
      % ACC
      ACC = array2table([ACC;mean(ACC,1)]);
      ACC.Properties.VariableNames = method_names;
      ACC.Properties.RowNames = test_names;
      
      % AFR
      AFR = array2table([AFR;mean(AFR,1)]);
      AFR.Properties.VariableNames = method_names;
      AFR.Properties.RowNames = test_names;
      
      % F1
      F1 = array2table([F1;mean(F1,1)]);
      F1.Properties.VariableNames = method_names;
      F1.Properties.RowNames = test_names;
      
      % MCC
      MCC = array2table([MCC;mean(MCC,1)]);
      MCC.Properties.VariableNames = method_names;
      MCC.Properties.RowNames = test_names;
      
      REPORT = [TPR('MEAN',:).Variables; TNR('MEAN',:).Variables;
        FPR('MEAN',:).Variables; FNR('MEAN',:).Variables;
        ACC('MEAN',:).Variables; AFR('MEAN',:).Variables;
        F1('MEAN',:).Variables; MCC('MEAN',:).Variables];
      
      REPORT = array2table(round(REPORT,4));
      REPORT.Properties.VariableNames = method_names;
      REPORT.Properties.RowNames = {'TPR','TNR','FPR','FNR','ACC','AFR','F1','MCC'};
      
      save(strcat(model_dir,'/report_results.mat'),'TPR','TNR',...
        'FPR','FNR','ACC','AFR','F1','MCC','REPORT');
      writetable(REPORT,strcat(model_dir,'/report_results.csv'),...
        'WriteRowNames',true,'Delimiter',',');
      fprintf('done!\n');
    end
    
    function reportTestResultsForKnnMethods(obj,model_dir)
      % ----------------------------------------------------------------------------------
      % Carrega e processa as métricas de acurácia dos métodos knn,
      % lmnn e klmnn para vários parâmetros de K e kappa
      % ----------------------------------------------------------------------------------      
      fprintf('\nComputing test results... ');
      % KNN
      KNN.TPR = nan*zeros(5,4); KNN.TNR = nan*zeros(5,4);
      KNN.FPR = nan*zeros(5,4); KNN.FNR = nan*zeros(5,4);
      KNN.ACC = nan*zeros(5,4); KNN.AFR = nan*zeros(5,4);
      KNN.F1 = nan*zeros(5,4);  KNN.MCC = nan*zeros(5,4);
      % LMNN
      LMNN.TPR = nan*zeros(5,4); LMNN.TNR = nan*zeros(5,4);
      LMNN.FPR = nan*zeros(5,4); LMNN.FNR = nan*zeros(5,4);
      LMNN.ACC = nan*zeros(5,4); LMNN.AFR = nan*zeros(5,4);
      LMNN.F1 = nan*zeros(5,4);  LMNN.MCC = nan*zeros(5,4);
      % KLMNN
      KLMNN.TPR = nan*zeros(5,4); KLMNN.TNR = nan*zeros(5,4);
      KLMNN.FPR = nan*zeros(5,4); KLMNN.FNR = nan*zeros(5,4);
      KLMNN.ACC = nan*zeros(5,4); KLMNN.AFR = nan*zeros(5,4);
      KLMNN.F1 = nan*zeros(5,4);  KLMNN.MCC = nan*zeros(5,4);
      
      % Testes
      for K = 1:5
        for kappa = K:-1:K-3
          if kappa >= 1
            j = K-kappa+1;
            knn_dir = strcat(model_dir,'/K=',int2str(K),' kappa=',int2str(kappa));
            fprintf('\nK = %d \tkappa = %d\n',K,kappa);
            % KNN
            knn_evaluation_file = strcat(knn_dir,'/knn_evaluations.mat');
            try
              knn_file = load(knn_evaluation_file);
              results = knn_file.knn_evaluations.results;
              KNN.TPR(K,j) = mean(results.TPR);
              KNN.TNR(K,j) = mean(results.TNR);
              KNN.FPR(K,j) = mean(results.FPR);
              KNN.FNR(K,j) = mean(results.FNR);
              KNN.AFR(K,j) = mean(results.AFR);
              KNN.ACC(K,j) = mean(results.ACC);
              KNN.F1(K,j) = mean(results.F1);
              KNN.MCC(K,j) = mean(results.MCC);
            catch
              fprintf('\n--> error knn results!\n');
            end
            
            % LMNN
            lmnn_evaluation_file = strcat(knn_dir,'/lmnn_evaluations.mat');
            try
              lmnn_file = load(lmnn_evaluation_file);
              results = lmnn_file.lmnn_evaluations.results;
              LMNN.TPR(K,j) = mean(results.TPR);
              LMNN.TNR(K,j) = mean(results.TNR);
              LMNN.FPR(K,j) = mean(results.FPR);
              LMNN.FNR(K,j) = mean(results.FNR);
              LMNN.AFR(K,j) = mean(results.AFR);
              LMNN.ACC(K,j) = mean(results.ACC);
              LMNN.F1(K,j) = mean(results.F1);
              LMNN.MCC(K,j) = mean(results.MCC);
            catch
              fprintf('\n--> error lmnn results!\n');
            end
            
            % KLMNN
            klmnn_evaluation_file = strcat(knn_dir,'/klmnn_evaluations.mat');
            try
              klmnn_file = load(klmnn_evaluation_file);
              results = klmnn_file.klmnn_evaluations.results;
              KLMNN.TPR(K,j) = mean(results.TPR);
              KLMNN.TNR(K,j) = mean(results.TNR);
              KLMNN.FPR(K,j) = mean(results.FPR);
              KLMNN.FNR(K,j) = mean(results.FNR);
              KLMNN.AFR(K,j) = mean(results.AFR);
              KLMNN.ACC(K,j) = mean(results.ACC);
              KLMNN.F1(K,j) = mean(results.F1);
              KLMNN.MCC(K,j) = mean(results.MCC);
            catch
              fprintf('\n--> error klmnn results!\n');
            end
          end
        end
      end
      
      % Cria as tabelas
      K_names = split(sprintf('K = %d,',1:5),',');
      K_names = K_names(1:end-1)';
      kappa_names = {'kappa1','kappa2','kappa3','kappa4'};
      
      kappa = struct('kappa1','K','kappa2','K-1','kappa3','K-2','kappa4','K-3');
      
      % KNN
      KNN.TPR = array2table(round(KNN.TPR,4),'VariableNames',kappa_names,'RowNames',K_names);
      KNN.TNR = array2table(round(KNN.TNR,4),'VariableNames',kappa_names,'RowNames',K_names);
      KNN.FPR = array2table(round(KNN.FPR,4),'VariableNames',kappa_names,'RowNames',K_names);
      KNN.FNR = array2table(round(KNN.FNR,4),'VariableNames',kappa_names,'RowNames',K_names);
      KNN.ACC = array2table(round(KNN.ACC,4),'VariableNames',kappa_names,'RowNames',K_names);
      KNN.AFR = array2table(round(KNN.AFR,4),'VariableNames',kappa_names,'RowNames',K_names);
      KNN.F1 = array2table(round(KNN.F1,4),'VariableNames',kappa_names,'RowNames',K_names);
      KNN.MCC = array2table(round(KNN.MCC,4),'VariableNames',kappa_names,'RowNames',K_names);
      
      % LMNN
      LMNN.TPR = array2table(round(LMNN.TPR,4),'VariableNames',kappa_names,'RowNames',K_names);
      LMNN.TNR = array2table(round(LMNN.TNR,4),'VariableNames',kappa_names,'RowNames',K_names);
      LMNN.FPR = array2table(round(LMNN.FPR,4),'VariableNames',kappa_names,'RowNames',K_names);
      LMNN.FNR = array2table(round(LMNN.FNR,4),'VariableNames',kappa_names,'RowNames',K_names);
      LMNN.ACC = array2table(round(LMNN.ACC,4),'VariableNames',kappa_names,'RowNames',K_names);
      LMNN.AFR = array2table(round(LMNN.AFR,4),'VariableNames',kappa_names,'RowNames',K_names);
      LMNN.F1 = array2table(round(LMNN.F1,4),'VariableNames',kappa_names,'RowNames',K_names);
      LMNN.MCC = array2table(round(LMNN.MCC,4),'VariableNames',kappa_names,'RowNames',K_names);
      
      % KLMNN
      KLMNN.TPR = array2table(round(KLMNN.TPR,4),'VariableNames',kappa_names,'RowNames',K_names);
      KLMNN.TNR = array2table(round(KLMNN.TNR,4),'VariableNames',kappa_names,'RowNames',K_names);
      KLMNN.FPR = array2table(round(KLMNN.FPR,4),'VariableNames',kappa_names,'RowNames',K_names);
      KLMNN.FNR = array2table(round(KLMNN.FNR,4),'VariableNames',kappa_names,'RowNames',K_names);
      KLMNN.ACC = array2table(round(KLMNN.ACC,4),'VariableNames',kappa_names,'RowNames',K_names);
      KLMNN.AFR = array2table(round(KLMNN.AFR,4),'VariableNames',kappa_names,'RowNames',K_names);
      KLMNN.F1 = array2table(round(KLMNN.F1,4),'VariableNames',kappa_names,'RowNames',K_names);
      KLMNN.MCC = array2table(round(KLMNN.MCC,4),'VariableNames',kappa_names,'RowNames',K_names);
      
      save(strcat(model_dir,'/report_knn_obj.method.mat'),'kappa','KNN','LMNN','KLMNN');
      fprintf('done!\n');
    end
    
    function reportEvaluationForKnnMethods(obj,model_dir)
      % ----------------------------------------------------------------------------------
      % Carrega e processa as métricas de acurácia dos métodos knn,
      % lmnn e klmnn para vários parâmetros de K e kappa
      % ----------------------------------------------------------------------------------      
      fprintf('\nComputing test results... ');
      % KNN
      KNN.TPR = nan*zeros(5,4); KNN.TNR = nan*zeros(5,4);
      KNN.FPR = nan*zeros(5,4); KNN.FNR = nan*zeros(5,4);
      KNN.ACC = nan*zeros(5,4); KNN.AFR = nan*zeros(5,4);
      KNN.F1 = nan*zeros(5,4);  KNN.MCC = nan*zeros(5,4);
      % LMNN
      LMNN.TPR = nan*zeros(5,4); LMNN.TNR = nan*zeros(5,4);
      LMNN.FPR = nan*zeros(5,4); LMNN.FNR = nan*zeros(5,4);
      LMNN.ACC = nan*zeros(5,4); LMNN.AFR = nan*zeros(5,4);
      LMNN.F1 = nan*zeros(5,4);  LMNN.MCC = nan*zeros(5,4);
      % KLMNN
      KLMNN.TPR = nan*zeros(5,4); KLMNN.TNR = nan*zeros(5,4);
      KLMNN.FPR = nan*zeros(5,4); KLMNN.FNR = nan*zeros(5,4);
      KLMNN.ACC = nan*zeros(5,4); KLMNN.AFR = nan*zeros(5,4);
      KLMNN.F1 = nan*zeros(5,4);  KLMNN.MCC = nan*zeros(5,4);
      
      % Testes
      for K = 1:5
        for kappa = K:-1:K-3
          if kappa >= 1
            j = K-kappa+1;
            knn_dir = strcat(model_dir,'/K=',int2str(K),' kappa=',int2str(kappa));
            fprintf('\nK = %d \tkappa = %d\n',K,kappa);
            % KNN
            knn_evaluation_file = strcat(knn_dir,'/knn_evaluation.mat');
            try
              knn_evaluation = load(knn_evaluation_file);
              KNN.TPR(K,j) = knn_evaluation.TPR;
              KNN.TNR(K,j) = knn_evaluation.TNR;
              KNN.FPR(K,j) = knn_evaluation.FPR;
              KNN.FNR(K,j) = knn_evaluation.FNR;
              KNN.AFR(K,j) = knn_evaluation.AFR;
              KNN.ACC(K,j) = knn_evaluation.ACC;
              KNN.F1(K,j) = knn_evaluation.F1;
              KNN.MCC(K,j) = knn_evaluation.MCC;
            catch
              fprintf('\n--> error knn results!\n');
            end
            
            % LMNN
            lmnn_evaluation_file = strcat(knn_dir,'/lmnn_evaluation.mat');
            try
              lmnn_evaluation = load(lmnn_evaluation_file);
              LMNN.TPR(K,j) = lmnn_evaluation.TPR;
              LMNN.TNR(K,j) = lmnn_evaluation.TNR;
              LMNN.FPR(K,j) = lmnn_evaluation.FPR;
              LMNN.FNR(K,j) = lmnn_evaluation.FNR;
              LMNN.AFR(K,j) = lmnn_evaluation.AFR;
              LMNN.ACC(K,j) = lmnn_evaluation.ACC;
              LMNN.F1(K,j) = lmnn_evaluation.F1;
              LMNN.MCC(K,j) = lmnn_evaluation.MCC;
            catch
              fprintf('\n--> error lmnn results!\n');
            end
            
            % KLMNN
            klmnn_evaluation_file = strcat(knn_dir,'/klmnn_evaluation.mat');
            try
              klmnn_evaluation = load(klmnn_evaluation_file);
              KLMNN.TPR(K,j) = klmnn_evaluation.TPR;
              KLMNN.TNR(K,j) = klmnn_evaluation.TNR;
              KLMNN.FPR(K,j) = klmnn_evaluation.FPR;
              KLMNN.FNR(K,j) = klmnn_evaluation.FNR;
              KLMNN.AFR(K,j) = klmnn_evaluation.AFR;
              KLMNN.ACC(K,j) = klmnn_evaluation.ACC;
              KLMNN.F1(K,j) = klmnn_evaluation.F1;
              KLMNN.MCC(K,j) = klmnn_evaluation.MCC;
            catch
              fprintf('\n--> error klmnn results!\n');
            end
          end
        end
      end
      
      % Cria as tabelas
      K_names = split(sprintf('K = %d,',1:5),',');
      K_names = K_names(1:end-1)';
      kappa_names = {'kappa1','kappa2','kappa3','kappa4'};
      
      kappa = struct('kappa1','K','kappa2','K-1','kappa3','K-2','kappa4','K-3');
      
      % KNN
      KNN.TPR = array2table(round(KNN.TPR,2),'VariableNames',kappa_names,'RowNames',K_names);
      KNN.TNR = array2table(round(KNN.TNR,2),'VariableNames',kappa_names,'RowNames',K_names);
      KNN.FPR = array2table(round(KNN.FPR,2),'VariableNames',kappa_names,'RowNames',K_names);
      KNN.FNR = array2table(round(KNN.FNR,2),'VariableNames',kappa_names,'RowNames',K_names);
      KNN.ACC = array2table(round(KNN.ACC,2),'VariableNames',kappa_names,'RowNames',K_names);
      KNN.AFR = array2table(round(KNN.AFR,2),'VariableNames',kappa_names,'RowNames',K_names);
      KNN.F1 = array2table(round(KNN.F1,2),'VariableNames',kappa_names,'RowNames',K_names);
      KNN.MCC = array2table(round(KNN.MCC,2),'VariableNames',kappa_names,'RowNames',K_names);
      
      % LMNN
      LMNN.TPR = array2table(round(LMNN.TPR,2),'VariableNames',kappa_names,'RowNames',K_names);
      LMNN.TNR = array2table(round(LMNN.TNR,2),'VariableNames',kappa_names,'RowNames',K_names);
      LMNN.FPR = array2table(round(LMNN.FPR,2),'VariableNames',kappa_names,'RowNames',K_names);
      LMNN.FNR = array2table(round(LMNN.FNR,2),'VariableNames',kappa_names,'RowNames',K_names);
      LMNN.ACC = array2table(round(LMNN.ACC,2),'VariableNames',kappa_names,'RowNames',K_names);
      LMNN.AFR = array2table(round(LMNN.AFR,2),'VariableNames',kappa_names,'RowNames',K_names);
      LMNN.F1 = array2table(round(LMNN.F1,2),'VariableNames',kappa_names,'RowNames',K_names);
      LMNN.MCC = array2table(round(LMNN.MCC,2),'VariableNames',kappa_names,'RowNames',K_names);
      
      % KLMNN
      KLMNN.TPR = array2table(round(KLMNN.TPR,2),'VariableNames',kappa_names,'RowNames',K_names);
      KLMNN.TNR = array2table(round(KLMNN.TNR,2),'VariableNames',kappa_names,'RowNames',K_names);
      KLMNN.FPR = array2table(round(KLMNN.FPR,2),'VariableNames',kappa_names,'RowNames',K_names);
      KLMNN.FNR = array2table(round(KLMNN.FNR,2),'VariableNames',kappa_names,'RowNames',K_names);
      KLMNN.ACC = array2table(round(KLMNN.ACC,2),'VariableNames',kappa_names,'RowNames',K_names);
      KLMNN.AFR = array2table(round(KLMNN.AFR,2),'VariableNames',kappa_names,'RowNames',K_names);
      KLMNN.F1 = array2table(round(KLMNN.F1,2),'VariableNames',kappa_names,'RowNames',K_names);
      KLMNN.MCC = array2table(round(KLMNN.MCC,2),'VariableNames',kappa_names,'RowNames',K_names);
      
      save(strcat(model_dir,'/report_knn_obj.method.mat'),'kappa','KNN','LMNN','KLMNN');
      fprintf('done!\n');
    end
    
    function reportExecutionTimeAndMetrics(obj,model_dir,N,DIM)
      % ----------------------------------------------------------------------------------
      % Carrega e processa o experimento de tempo de execução e avalia métricas
      % ----------------------------------------------------------------------------------      
      % Eobj.XPERIMENTO 1 (QUANTIDADE DE DADOS DE TREINO)
      exp1.MCC = zeros(7,numel(N));
      exp1.F1 = zeros(7,numel(N));
      exp1.val_time = zeros(7,numel(N));
      exp1.mean_test_time = zeros(7,numel(N));
      % Variação do número de exemplos de treino
      for j=1:numel(N)
        dim = 10;
        exp_dir = strcat(model_dir,'/N=',int2str(N(j)),' DIM=',int2str(dim));
        for i=1:numel(obj.method)
          knn_dir = strcat(exp_dir,'/K=',int2str(obj.knn_arg),' kappa=',int2str(obj.knn_threshold));
          switch obj.method{i}
            case 'knn'
              try
                file_model = strcat(knn_dir,'/knn_model.mat');
                model = load(file_model);
                exp1.val_time(1,j) = model.knn_model.validation_time;
                
                file_test = strcat(knn_dir,'/knn_evaluation.mat');
                tests = load(file_test);
                exp1.MCC(1,j) = tests.MCC;
                exp1.F1(1,j) = tests.F1;
                exp1.test_time(1,j) = tests.evaluation_time;
              catch
                fprintf('\n--> error loading knn!\n');
              end
            case 'lmnn'
              try
                file_model = strcat(knn_dir,'/lmnn_model.mat');
                model = load(file_model);
                exp1.val_time(2,j) = model.lmnn_model.validation_time;
                
                file_test = strcat(knn_dir,'/lmnn_evaluation.mat');
                tests = load(file_test);
                exp1.MCC(2,j) = tests.MCC;
                exp1.F1(2,j) = tests.F1;
                exp1.test_time(2,j) = tests.evaluation_time;
              catch
                fprintf('\n--> error loading lmnn!\n');
              end
            case 'klmnn'
              try
                file_model = strcat(knn_dir,'/klmnn_model.mat');
                model = load(file_model);
                exp1.val_time(3,j) = model.klmnn_model.validation_time;
                
                file_test = strcat(knn_dir,'/klmnn_evaluation.mat');
                tests = load(file_test);
                exp1.MCC(3,j) = tests.MCC;
                exp1.F1(3,j) = tests.F1;
                exp1.test_time(3,j) = tests.evaluation_time;
              catch
                fprintf('\n--> error loading klmnn!\n');
              end
            case 'knfst'
              try
                file_model = strcat(exp_dir,'/knfst_model.mat');
                model = load(file_model);
                exp1.val_time(4,j) = model.knfst_model.validation_time;
                
                file_test = strcat(exp_dir,'/knfst_evaluation.mat');
                tests = load(file_test);
                exp1.MCC(4,j) = tests.MCC;
                exp1.F1(4,j) = tests.F1;
                exp1.test_time(4,j) = tests.evaluation_time;
              catch
                fprintf('\n--> error loading knfst!\n');
              end
            case 'one_svm'
              try
                file_model = strcat(exp_dir,'/one_svm_model.mat');
                model = load(file_model);
                exp1.val_time(5,j) = model.one_svm_model.validation_time;
                
                file_test = strcat(exp_dir,'/one_svm_evaluation.mat');
                tests = load(file_test);
                exp1.MCC(5,j) = tests.MCC;
                exp1.F1(5,j) = tests.F1;
                exp1.test_time(5,j) = tests.evaluation_time;
              catch
                fprintf('\n--> error loading one svm!\n');
              end
            case 'multi_svm'
              try
                file_model = strcat(exp_dir,'/multi_svm_model.mat');
                model = load(file_model);
                exp1.val_time(6,j) = model.multi_svm_model.validation_time;
                
                file_test = strcat(exp_dir,'/multi_svm_evaluation.mat');
                tests = load(file_test);
                exp1.MCC(6,j) = tests.MCC;
                exp1.F1(6,j) = tests.F1;
                exp1.test_time(6,j) = tests.evaluation_time;
              catch
                fprintf('\n--> error loading multi svm!\n');
              end
            case 'kpca'
              try
                file_model = strcat(exp_dir,'/kpca_model.mat');
                model = load(file_model);
                exp1.val_time(7,j) = model.kpca_model.validation_time;
                
                file_test = strcat(exp_dir,'/kpca_evaluation.mat');
                tests = load(file_test);
                exp1.MCC(7,j) = tests.MCC;
                exp1.F1(7,j) = tests.F1;
                exp1.test_time(7,j) = tests.evaluation_time;
              catch
                fprintf('\n--> error loading kpca nov!\n');
              end
          end
        end
      end
      var_names = split(sprintf('N%d,',N),',');
      var_names = var_names(1:end-1);
      row_names = {'KNN','LMNN','KLMNN','KNFST','ONE_SVM','MULTI_SVM','KPCA_NOV'};
      
      exp1.MCC = array2table(exp1.MCC,'VariableNames',var_names,'RowNames',row_names);
      exp1.F1 = array2table(exp1.F1,'VariableNames',var_names,'RowNames',row_names);
      exp1.val_time = array2table(exp1.val_time,...
        'VariableNames',var_names,'RowNames',row_names);
      exp1.mean_test_time = array2table(exp1.mean_test_time,...
        'VariableNames',var_names,'RowNames',row_names);
      
      % Eobj.XPERIMENTO 2 (DIMENSÃO)
      exp2.MCC = zeros(7,numel(DIM));
      exp2.F1 = zeros(7,numel(DIM));
      exp2.val_time = zeros(7,numel(DIM));
      exp2.mean_test_time = zeros(7,numel(DIM));
      % Variação do número de exemplos de treino
      for j=1:numel(DIM)
        n = 400;
        exp_dir = strcat(model_dir,'/N=',int2str(n),' DIM=',int2str(DIM(j)));
        for i=1:numel(obj.method)
          knn_dir = strcat(exp_dir,'/K=',int2str(obj.knn_arg),' kappa=',int2str(obj.knn_threshold));
          switch obj.method{i}
            case 'knn'
              try
                file_model = strcat(knn_dir,'/knn_model.mat');
                model = load(file_model);
                exp2.val_time(1,j) = model.knn_model.validation_time;
                
                file_test = strcat(knn_dir,'/knn_evaluation.mat');
                tests = load(file_test);
                exp2.MCC(1,j) = tests.MCC;
                exp2.F1(1,j) = tests.F1;
                exp2.test_time(1,j) = tests.evaluation_time;
              catch
                fprintf('\n--> error loading knn!\n');
              end
            case 'lmnn'
              try
                file_model = strcat(knn_dir,'/lmnn_model.mat');
                model = load(file_model);
                exp2.val_time(2,j) = model.lmnn_model.validation_time;
                
                file_test = strcat(knn_dir,'/lmnn_evaluation.mat');
                tests = load(file_test);
                exp2.MCC(2,j) = tests.MCC;
                exp2.F1(2,j) = tests.F1;
                exp2.test_time(2,j) = tests.evaluation_time;
              catch
                fprintf('\n--> error loading lmnn!\n');
              end
            case 'klmnn'
              try
                file_model = strcat(knn_dir,'/klmnn_model.mat');
                model = load(file_model);
                exp2.val_time(3,j) = model.klmnn_model.validation_time;
                
                file_test = strcat(knn_dir,'/klmnn_evaluation.mat');
                tests = load(file_test);
                exp2.MCC(3,j) = tests.MCC;
                exp2.F1(3,j) = tests.F1;
                exp2.test_time(3,j) = tests.evaluation_time;
              catch
                fprintf('\n--> error loading klmnn!\n');
              end
            case 'knfst'
              try
                file_model = strcat(exp_dir,'/knfst_model.mat');
                model = load(file_model);
                exp2.val_time(4,j) = model.knfst_model.validation_time;
                
                file_test = strcat(exp_dir,'/knfst_evaluation.mat');
                tests = load(file_test);
                exp2.MCC(4,j) = tests.MCC;
                exp2.F1(4,j) = tests.F1;
                exp2.test_time(4,j) = tests.evaluation_time;
              catch
                fprintf('\n--> error loading knfst!\n');
              end
            case 'one_svm'
              try
                file_model = strcat(exp_dir,'/one_svm_model.mat');
                model = load(file_model);
                exp2.val_time(5,j) = model.one_svm_model.validation_time;
                
                file_test = strcat(exp_dir,'/one_svm_evaluation.mat');
                tests = load(file_test);
                exp2.MCC(5,j) = tests.MCC;
                exp2.F1(5,j) = tests.F1;
                exp2.test_time(5,j) = tests.evaluation_time;
              catch
                fprintf('\n--> error loading one svm!\n');
              end
            case 'multi_svm'
              try
                file_model = strcat(exp_dir,'/multi_svm_model.mat');
                model = load(file_model);
                exp2.val_time(6,j) = model.multi_svm_model.validation_time;
                
                file_test = strcat(exp_dir,'/multi_svm_evaluation.mat');
                tests = load(file_test);
                exp2.MCC(6,j) = tests.MCC;
                exp2.F1(6,j) = tests.F1;
                exp2.test_time(6,j) = tests.evaluation_time;
              catch
                fprintf('\n--> error loading multi svm!\n');
              end
            case 'kpca'
              try
                file_model = strcat(exp_dir,'/kpca_model.mat');
                model = load(file_model);
                exp2.val_time(7,j) = model.kpca_model.validation_time;
                
                file_test = strcat(exp_dir,'/kpca_evaluation.mat');
                tests = load(file_test);
                exp2.MCC(7,j) = tests.MCC;
                exp2.F1(7,j) = tests.F1;
                exp2.test_time(7,j) = tests.evaluation_time;
              catch
                fprintf('\n--> error loading kpca nov!\n');
              end
          end
        end
      end
      var_names = split(sprintf('D%d,',DIM),',');
      var_names = var_names(1:end-1);
      row_names = {'KNN','LMNN','KLMNN','KNFST','ONE_SVM','MULTI_SVM','KPCA_NOV'};
      
      exp2.MCC = array2table(exp2.MCC,'VariableNames',var_names,'RowNames',row_names);
      exp2.F1 = array2table(exp2.F1,'VariableNames',var_names,'RowNames',row_names);
      exp2.val_time = array2table(exp2.val_time,...
        'VariableNames',var_names,'RowNames',row_names);
      exp2.mean_test_time = array2table(exp2.mean_test_time,...
        'VariableNames',var_names,'RowNames',row_names);
      
      % PLOTS Eobj.XPERIMENTO 1
      figure;
      clf('reset');
      hold on;
      plot(N,exp1.MCC.Variables','-s','LineWidth',1);
      hold off;
      title('Metric');
      xlabel('# training samples');
      ylabel('matthews correlation coefficient (mcc)');
      legend({'knn','lmnn','klmnn','knfst','one svm','multi svm','kpca nov'});
      
      figure;
      clf('reset');
      hold on;
      plot(N,exp1.test_time.Variables/60','-s','LineWidth',1);
      hold off;
      title('Test time');
      xlabel('# training samples');
      ylabel('evaluation time (minutes)');
      legend({'knn','lmnn','klmnn','knfst','one svm','multi svm','kpca nov'});
      
      figure;
      clf('reset');
      hold on;
      plot(N,exp1.val_time.Variables/3600','-s','LineWidth',1);
      hold off;
      title('Validation time');
      xlabel('# training samples');
      ylabel('validation time (hours)');
      legend({'knn','lmnn','klmnn','knfst','one svm','multi svm','kpca nov'});
      
      figure;
      clf('reset');
      hold on;
      plot(DIM,exp2.MCC.Variables','-s','LineWidth',1);
      hold off;
      title('Metric');
      xlabel('# dimensions');
      ylabel('matthews correlation coefficient (mcc)');
      legend({'knn','lmnn','klmnn','knfst','one svm','multi svm','kpca nov'});
      
      figure;
      clf('reset');
      hold on;
      plot(DIM,exp2.test_time.Variables/60','-s','LineWidth',1);
      hold off;
      title('Test time');
      xlabel('# dimensions');
      ylabel('evaluation time (minutes)');
      legend({'knn','lmnn','klmnn','knfst','one svm','multi svm','kpca nov'});
      
      figure;
      clf('reset');
      hold on;
      plot(DIM,exp2.val_time.Variables/3600','-s','LineWidth',1);
      hold off;
      title('Validation time');
      xlabel('# dimensions');
      ylabel('validation time (hours)');
      legend({'knn','lmnn','klmnn','knfst','one svm','multi svm','kpca nov'});
      
      save(strcat(model_dir,'/report_experiment.mat'),'exp1','exp2');
    end
    
    function reportExecutionTimeAndMetricsTests(obj,model_dir,N,DIM)
      % ----------------------------------------------------------------------------------
      % Carrega e processa o experimento de tempo de execução e avalia métricas
      % ----------------------------------------------------------------------------------
      exp1_file = strcat(model_dir,'/report_experiment_1.mat');
      if ~exist(exp1_file,'file')
        % Eobj.XPERIMENTO 1 (QUANTIDADE DE DADOS DE TREINO)
        fprintf('Processing results for the training sample variation experiment... ');
        exp1.MCC = zeros(7,numel(N));
        exp1.F1 = zeros(7,numel(N));
        exp1.val_time = zeros(7,numel(N));
        exp1.mean_test_time = zeros(7,numel(N));
        % Variação do número de exemplos de treino
        for j=1:numel(N)
          dim = 10;
          exp_dir = strcat(model_dir,'/N=',int2str(N(j)),' DIM=',int2str(dim));
          for i=1:numel(obj.method)
            knn_dir = strcat(exp_dir,'/K=',int2str(obj.knn_arg),' kappa=',int2str(obj.knn_threshold));
            switch obj.method{i}
              case 'knn'
                try
                  file_model = strcat(knn_dir,'/knn_model.mat');
                  model = load(file_model);
                  exp1.val_time(1,j) = model.knn_model.validation_time;
                  
                  file_test = strcat(knn_dir,'/knn_evaluation_tests.mat');
                  load(file_test,'results','evaluation_time');
                  exp1.MCC(1,j) = mean(results.MCC);
                  exp1.F1(1,j) = mean(results.F1);
                  exp1.mean_test_time(1,j) = evaluation_time/size(results,1);
                catch
                  fprintf('\n--> error processing  knn!\n');
                end
              case 'lmnn'
                try
                  file_model = strcat(knn_dir,'/lmnn_model.mat');
                  model = load(file_model);
                  exp1.val_time(2,j) = model.lmnn_model.validation_time;
                  
                  file_test = strcat(knn_dir,'/lmnn_evaluation_tests.mat');
                  load(file_test,'results','evaluation_time');
                  exp1.MCC(2,j) = mean(results.MCC);
                  exp1.F1(2,j) = mean(results.F1);
                  exp1.mean_test_time(2,j) = evaluation_time/size(results,1);
                catch
                  fprintf('\n--> error processing  lmnn!\n');
                end
              case 'klmnn'
                try
                  file_model = strcat(knn_dir,'/klmnn_model.mat');
                  model = load(file_model);
                  exp1.val_time(3,j) = model.klmnn_model.validation_time;
                  
                  file_test = strcat(knn_dir,'/klmnn_evaluation_tests.mat');
                  load(file_test,'results','evaluation_time');
                  exp1.MCC(3,j) = mean(results.MCC);
                  exp1.F1(3,j) = mean(results.F1);
                  exp1.mean_test_time(3,j) = evaluation_time/size(results,1);
                catch
                  fprintf('\n--> error processing  klmnn!\n');
                end
              case 'knfst'
                try
                  file_model = strcat(exp_dir,'/knfst_model.mat');
                  model = load(file_model);
                  exp1.val_time(4,j) = model.knfst_model.validation_time;
                  
                  file_test = strcat(exp_dir,'/knfst_evaluation_tests.mat');
                  load(file_test,'results','evaluation_time');
                  exp1.MCC(4,j) = mean(results.MCC);
                  exp1.F1(4,j) = mean(results.F1);
                  exp1.mean_test_time(4,j) = evaluation_time/size(results,1);
                catch
                  fprintf('\n--> error processing  knfst!\n');
                end
              case 'one_svm'
                try
                  file_model = strcat(exp_dir,'/one_svm_model.mat');
                  model = load(file_model);
                  exp1.val_time(5,j) = model.one_svm_model.validation_time;
                  
                  file_test = strcat(exp_dir,'/one_svm_evaluation_tests.mat');
                  load(file_test,'results','evaluation_time');
                  exp1.MCC(5,j) = mean(results.MCC);
                  exp1.F1(5,j) = mean(results.F1);
                  exp1.mean_test_time(5,j) = evaluation_time/size(results,1);
                catch
                  fprintf('\n--> error processing  one svm!\n');
                end
              case 'multi_svm'
                try
                  file_model = strcat(exp_dir,'/multi_svm_model.mat');
                  model = load(file_model);
                  exp1.val_time(6,j) = model.multi_svm_model.validation_time;
                  
                  file_test = strcat(exp_dir,'/multi_svm_evaluation_tests.mat');
                  load(file_test,'results','evaluation_time');
                  exp1.MCC(6,j) = mean(results.MCC);
                  exp1.F1(6,j) = mean(results.F1);
                  exp1.mean_test_time(6,j) = evaluation_time/size(results,1);
                catch
                  fprintf('\n--> error processing  multi svm!\n');
                end
              case 'kpca'
                try
                  file_model = strcat(exp_dir,'/kpca_model.mat');
                  model = load(file_model);
                  exp1.val_time(7,j) = model.kpca_model.validation_time;
                  
                  file_test = strcat(exp_dir,'/kpca_evaluation_tests.mat');
                  load(file_test,'results','evaluation_time');
                  exp1.MCC(7,j) = mean(results.MCC);
                  exp1.F1(7,j) = mean(results.F1);
                  exp1.mean_test_time(7,j) = evaluation_time/size(results,1);
                catch
                  fprintf('\n--> error processing  kpca!\n');
                end
            end
          end
        end
        var_names = split(sprintf('N%d,',N),',');
        var_names = var_names(1:end-1);
        row_names = {'KNN','LMNN','KLMNN','KNFST','ONE_SVM','MULTI_SVM','KPCA_NOV'};
        
        exp1.MCC = array2table(exp1.MCC,'VariableNames',var_names,'RowNames',row_names);
        exp1.F1 = array2table(exp1.F1,'VariableNames',var_names,'RowNames',row_names);
        exp1.val_time = array2table(exp1.val_time,...
          'VariableNames',var_names,'RowNames',row_names);
        exp1.mean_test_time = array2table(exp1.mean_test_time,...
          'VariableNames',var_names,'RowNames',row_names);
        
        fprintf('done!\n');
      else
        exp1 = load(exp1_file);
      end
      
      exp2_file = strcat(model_dir,'/report_experiment_2.mat');
      if ~exist(exp2_file,'file')
        fprintf('Processing results for the dimension variation experiment... ');
        % Eobj.XPERIMENTO 2 (DIMENSÃO)
        exp2.MCC = zeros(7,numel(DIM));
        exp2.F1 = zeros(7,numel(DIM));
        exp2.val_time = zeros(7,numel(DIM));
        exp2.mean_test_time = zeros(7,numel(DIM));
        % Variação do número de exemplos de treino
        for j=1:numel(DIM)
          n = 800;
          exp_dir = strcat(model_dir,'/N=',int2str(n),' DIM=',int2str(DIM(j)));
          for i=1:numel(obj.method)
            knn_dir = strcat(exp_dir,'/K=',int2str(obj.knn_arg),...
              ' kappa=',int2str(obj.knn_threshold));
            switch obj.method{i}
              case 'knn'
                try
                  file_model = strcat(knn_dir,'/knn_model.mat');
                  model = load(file_model);
                  exp2.val_time(1,j) = model.knn_model.validation_time;
                  
                  file_test = strcat(knn_dir,'/knn_evaluation_tests.mat');
                  load(file_test,'results','evaluation_time');
                  exp2.MCC(1,j) = mean(results.MCC);
                  exp2.F1(1,j) = mean(results.F1);
                  exp2.mean_test_time(1,j) = evaluation_time/size(results,1);
                catch
                  fprintf('\n--> error loading knn!\n');
                end
              case 'lmnn'
                try
                  file_model = strcat(knn_dir,'/lmnn_model.mat');
                  model = load(file_model);
                  exp2.val_time(2,j) = model.lmnn_model.validation_time;
                  
                  file_test = strcat(knn_dir,'/lmnn_evaluation_tests.mat');
                  load(file_test,'results','evaluation_time');
                  exp2.MCC(2,j) = mean(results.MCC);
                  exp2.F1(2,j) = mean(results.F1);
                  exp2.mean_test_time(2,j) = evaluation_time/size(results,1);
                catch
                  fprintf('\n--> error loading lmnn!\n');
                end
              case 'klmnn'
                try
                  file_model = strcat(knn_dir,'/klmnn_model.mat');
                  model = load(file_model);
                  exp2.val_time(3,j) = model.klmnn_model.validation_time;
                  
                  file_test = strcat(knn_dir,'/klmnn_evaluation_tests.mat');
                  load(file_test,'results','evaluation_time');
                  exp2.MCC(3,j) = mean(results.MCC);
                  exp2.F1(3,j) = mean(results.F1);
                  exp2.mean_test_time(3,j) = evaluation_time/size(results,1);
                catch
                  fprintf('\n--> error processing  klmnn!\n');
                end
              case 'knfst'
                try
                  file_model = strcat(exp_dir,'/knfst_model.mat');
                  model = load(file_model);
                  exp2.val_time(4,j) = model.knfst_model.validation_time;
                  
                  file_test = strcat(exp_dir,'/knfst_evaluation_tests.mat');
                  load(file_test,'results','evaluation_time');
                  exp2.MCC(4,j) = mean(results.MCC);
                  exp2.F1(4,j) = mean(results.F1);
                  exp2.mean_test_time(4,j) = evaluation_time/size(results,1);
                catch
                  fprintf('\n--> error processing knfst!\n');
                end
              case 'one_svm'
                try
                  file_model = strcat(exp_dir,'/one_svm_model.mat');
                  model = load(file_model);
                  exp2.val_time(5,j) = model.one_svm_model.validation_time;
                  
                  file_test = strcat(exp_dir,'/one_svm_evaluation_tests.mat');
                  load(file_test,'results','evaluation_time');
                  exp2.MCC(5,j) = mean(results.MCC);
                  exp2.F1(5,j) = mean(results.F1);
                  exp2.mean_test_time(5,j) = evaluation_time/size(results,1);
                catch
                  fprintf('\n--> error processing one svm!\n');
                end
              case 'multi_svm'
                try
                  file_model = strcat(exp_dir,'/multi_svm_model.mat');
                  model = load(file_model);
                  exp2.val_time(6,j) = model.multi_svm_model.validation_time;
                  
                  file_test = strcat(exp_dir,'/multi_svm_evaluation_tests.mat');
                  load(file_test,'results','evaluation_time');
                  exp2.MCC(6,j) = mean(results.MCC);
                  exp2.F1(6,j) = mean(results.F1);
                  exp2.mean_test_time(6,j) = evaluation_time/size(results,1);
                catch
                  fprintf('\n--> error processing multi svm!\n');
                end
              case 'kpca'
                try
                  file_model = strcat(exp_dir,'/kpca_model.mat');
                  model = load(file_model);
                  exp2.val_time(7,j) = model.kpca_model.validation_time;
                  
                  file_test = strcat(exp_dir,'/kpca_evaluation_tests.mat');
                  load(file_test,'results','evaluation_time');
                  exp2.MCC(7,j) = mean(results.MCC);
                  exp2.F1(7,j) = mean(results.F1);
                  exp2.mean_test_time(7,j) = evaluation_time/size(results,1);
                catch
                  fprintf('\n--> error loading kpca nov!\n');
                end
            end
          end
        end
        var_names = split(sprintf('D%d,',DIM),',');
        var_names = var_names(1:end-1);
        row_names = {'KNN','LMNN','KLMNN','KNFST','ONE_SVM','MULTI_SVM','KPCA_NOV'};
        
        exp2.MCC = array2table(exp2.MCC,'VariableNames',var_names,'RowNames',row_names);
        exp2.F1 = array2table(exp2.F1,'VariableNames',var_names,'RowNames',row_names);
        exp2.val_time= array2table(exp2.val_time,...
          'VariableNames',var_names,'RowNames',row_names);
        exp2.mean_test_time = array2table(exp2.mean_test_time,...
          'VariableNames',var_names,'RowNames',row_names);
        
        fprintf('done!\n');
      else
        exp2 = load(exp2_file);
      end
      
      % PLOTS Eobj.XPERIMENTO 1
      figure;
      clf('reset');
      %subplot(1,2,1);
      hold on;
      p = plot(N,exp1.MCC.Variables','-s','LineWidth',1);
      p(1).Marker = 'o'; p(2).Marker = 's'; p(3).Marker = 'd';
      p(4).Marker = '^'; p(5).Marker = 'v'; p(6).Marker = '<';
      p(7).Marker = '>';
      hold off;
      %title('Metric');
      xlabel('training size');
      ylabel('matthews correlation coefficient (mcc)');
      legend({'knn','lmnn','klmnn','knfst','one svm','multi svm','kpca nov'},...
        'fontname','times','fontsize',12,'location','southeast');
      set(gca,'fontname','times','fontsize',12);
      set(gcf,'Position',[100 100 600 400]);
      saveas(gcf,strcat(model_dir,'/sim4-exp1-mcc.pdf'));
      saveas(gcf,strcat(model_dir,'/sim4-exp1-mcc.fig'));
      
      figure;
      clf('reset');
      %subplot(1,2,2);
      hold on;
      p = plot(N,exp1.mean_test_time.Variables','-s','LineWidth',1);
      p(1).Marker = 'o'; p(2).Marker = 's'; p(3).Marker = 'd';
      p(4).Marker = '^'; p(5).Marker = 'p'; p(6).Marker = 'x';
      p(7).Marker = '*';
      hold off;
      %title('Test time');
      xlabel('training size');
      ylabel('evaluation time (seconds)');
      legend({'knn','lmnn','klmnn','knfst','one svm','multi svm','kpca nov'},...
        'fontname','times','fontsize',12,'location','northwest');
      set(gca,'fontname','times','fontsize',12);
      set(gcf,'Position',[100 100 600 400]);
      %saveas(gcf,strcat(model_dir,'/sim4-exp1-test_time.pdf'));
      %saveas(gcf,strcat(model_dir,'/sim4-exp1-test_time.fig'));
      
      figure;
      clf('reset');
      hold on;
      p = plot(N,exp1.val_time.Variables'/60,'-s','LineWidth',1);
      p(1).Marker = 'o'; p(2).Marker = 's'; p(3).Marker = 'd';
      p(4).Marker = '^'; p(5).Marker = 'p'; p(6).Marker = 'x';
      p(7).Marker = '*';
      hold off;
      %title('Validation time');
      xlabel('training size');
      ylabel('hyperparameter optimization time (minutes)');
      legend({'knn','lmnn','klmnn','knfst','one svm','multi svm','kpca nov'},...
        'fontname','times','fontsize',12,'location','northwest');
      set(gca,'fontname','times','fontsize',12);
      set(gcf,'Position',[100 100 600 400]);
      saveas(gcf,strcat(model_dir,'/sim4-exp1-val_time.pdf'));
      saveas(gcf,strcat(model_dir,'/sim4-exp1-val_time.fig'));
      
      figure;
      %sublplot(1,1,2);
      clf('reset');
      hold on;
      p = plot(DIM,exp2.MCC.Variables','-s','LineWidth',1);
      p(1).Marker = 'o'; p(2).Marker = 's'; p(3).Marker = 'd';
      p(4).Marker = '^'; p(5).Marker = 'p'; p(6).Marker = 'x';
      p(7).Marker = '*';
      hold off;
      %title('Metric');
      xlabel('dimensions');
      ylabel('matthews correlation coefficient (mcc)');
      legend({'knn','lmnn','klmnn','knfst','one svm','multi svm','kpca nov'},...
        'fontname','times','fontsize',15,'location','southeast');
      set(gca,'fontname','times','fontsize',16.5);
      set(gcf,'Position',[100 100 600 400]);
      %saveas(gcf,strcat(model_dir,'/sim4-exp2-mcc.pdf'));
      %saveas(gcf,strcat(model_dir,'/sim4-exp2-mcc.fig'));
      
      figure;
      clf('reset');
      %subplot(2,1,2);
      hold on;
      p = plot(DIM,exp2.mean_test_time.Variables','-s','LineWidth',1);
      p(1).Marker = 'o'; p(2).Marker = 's'; p(3).Marker = 'd';
      p(4).Marker = '^'; p(5).Marker = 'p'; p(6).Marker = 'x';
      p(7).Marker = '*';
      hold off;
      %title('Test time');
      xlabel('dimensions');
      ylabel('evaluation time (seconds)');
      legend({'knn','lmnn','klmnn','knfst','one svm','multi svm','kpca nov'},...
        'fontname','times','fontsize',12,'location','northwest');
      set(gca,'fontname','times','fontsize',12);
      set(gcf,'Position',[100 100 600 400]);
      %saveas(gcf,strcat(model_dir,'/sim4-exp2-test_time.pdf'));
      %saveas(gcf,strcat(model_dir,'/sim4-exp2-test_time.fig'));
      
      figure;
      clf('reset');
      hold on;
      p = plot(DIM,exp2.val_time.Variables'/60,'-s','LineWidth',1);
      p(1).Marker = 'o'; p(2).Marker = 's'; p(3).Marker = 'd';
      p(4).Marker = '^'; p(5).Marker = 'p'; p(6).Marker = 'x';
      p(7).Marker = '*';
      hold off;
      %title('Validation time');
      xlabel('dimensions');
      ylabel('hyperparameter optimization time (minutes)');
      legend({'knn','lmnn','klmnn','knfst','one svm','multi svm','kpca nov'},...
        'fontname','times','fontsize',12,'location','southeast');
      set(gca,'fontname','times','fontsize',12);
      set(gcf,'Position',[100 100 600 400]);
      saveas(gcf,strcat(model_dir,'/sim4-exp2-val_time.pdf'));
      saveas(gcf,strcat(model_dir,'/sim4-exp2-val_time.pdf'));
      
      %save(strcat(model_dir,'/report_experiment_1.mat'),'-struct','exp1');
      %save(strcat(model_dir,'/report_experiment_2.mat'),'-struct','exp2');
    end    
  end
end