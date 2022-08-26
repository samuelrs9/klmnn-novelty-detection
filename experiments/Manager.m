classdef Manager < handle
  % --------------------------------------------------------------------------------------
  % This class is used to manage the executions and reports of the novelty 
  % detection algorithms.
  %
  % Version 2.0, July 2022.
  % By Samuel Silva (samuelrs@usp.br).
  % --------------------------------------------------------------------------------------  
  properties
      X = [];                       % data samples [num_samples x dimension]
      y = [];                       % labels [num_samples x 1]
      method = [];                  % novelty detection method used     
      out_dir = [];                 % output directory
      num_experiments = [];         % number of validation experiments
      num_classes = [];             % number of classes
      num_untrained_classes = [];   % number of untrained classes
      training_ratio = [];          % sample training ratio
      random_select_classes = true; % enable/disable random selection of untrained classes            
      plot_metric = false;          % enable/disable metric plotting
  end
  
  methods
    function obj = Manager(X,y,out_dir,num_experiments,...
      num_untrained_classes,training_ratio,random_select_classes,plot_metric)
      % ----------------------------------------------------------------------------------
      % Constructor.
      %
      % Input args
      %   X: data samples [num_samples x dimension].
      %   y: labels [num_samples x 1].
      %   out_dir: output directory of validation experiments.
      %   num_experiments: number of validation experiments.
      %   num_untrained_classes: number of untrained classes, this can be used to simulate 
      %     novelty data in the dataset.
      %   training_ratio: training sample rate.
      %   random_select_classes: a boolean that enable/disable random selection of 
      %     untrained classes.
      %   plot_metric: a boolean that enable/disable metric plotting.
      % ----------------------------------------------------------------------------------
      obj.X = X;      
      obj.y = y;
      obj.out_dir = out_dir;
      obj.num_classes = numel(unique(y));      
      obj.num_experiments = num_experiments;      
      obj.num_untrained_classes = num_untrained_classes;
      obj.training_ratio = training_ratio;
      obj.random_select_classes = random_select_classes;
      obj.plot_metric = plot_metric;
    end
    
    function runExperiments(obj,methods)
      %-----------------------------------------------------------------------------------
      % This method runs validation experiments and hyperparameter search for the 
      % chosen algorithm.
      %
      % Input args
      %   methods: a cell array of structs with hyperparameters for novelty 
      %     detection methods.
      % ----------------------------------------------------------------------------------
      for i=1:numel(methods)
        switch methods{i}.name
          case 'knn'
            try
              % It creates the output directory
              knn_dir = strcat(obj.out_dir,'/K=',int2str(methods{i}.knn_arg),...
                ' kappa=',int2str(methods{i}.kappa_threshold));
              file = strcat(knn_dir,'/knn_experiments.mat');
              if exist(file,'file')
                fprintf('Experiment KNN (K = %d kappa = %d) already exists!\n',...
                  methods{i}.knn_arg,methods{i}.kappa_threshold)
                continue;
              end               
              if ~exist(knn_dir,'dir')
                mkdir(knn_dir);
              end              
              % It creates an object of class KnnND
              knn = KnnND(obj.X,obj.y);              
              % It starts experiments                      
              experiments = knn.runExperiments(...
                methods{i},...
                obj.num_experiments,...                
                obj.num_untrained_classes,...
                obj.training_ratio,...
                obj.random_select_classes,...
                obj.plot_metric);              
              % Save experiments
              save(file,'-struct','experiments');
            catch
              fprintf('\n knn experiment error!!! \n');
            end
          case 'lmnn'
            try
              % It creates the output directory
              knn_dir = strcat(obj.out_dir,'/K=',int2str(methods{i}.knn_arg),...
                ' kappa=',int2str(methods{i}.kappa_threshold));
              file = strcat(knn_dir,'/lmnn_experiments.mat');
              if exist(file,'file')
                fprintf('Experiment LMNN (K = %d kappa = %d) already exists!\n',...
                  methods{i}.knn_arg,methods{i}.kappa_threshold)
                continue;
              end                                          
              if ~exist(knn_dir,'dir')
                mkdir(knn_dir);
              end                            
              % It creates an object of class LmnnND
              lmnn = LmnnND(obj.X,obj.y);
              % It starts experiments              
              experiments = lmnn.runExperiments(...
                methods{i},...
                obj.num_experiments,...                
                obj.num_untrained_classes,...
                obj.training_ratio,...
                obj.random_select_classes,...
                obj.plot_metric);                        
              % Save experiments              
              save(file ,'-struct','experiments');
            catch
              fprintf('\n lmnn experiment error!!! \n');
            end
          case 'klmnn'
            try
              % It creates the output directory
              knn_dir = strcat(obj.out_dir,'/K=',int2str(methods{i}.knn_arg),...
                ' kappa=',int2str(methods{i}.kappa_threshold));
              file = strcat(knn_dir,'/klmnn_experiments.mat');
              if exist(file,'file')
                fprintf('Experiment KLMNN (K = %d kappa = %d) already exists!\n',...
                  methods{i}.knn_arg,methods{i}.kappa_threshold)
                continue;
              end               
              if ~exist(knn_dir,'dir')
                mkdir(knn_dir);
              end                            
              % It creates an object of class KlmnnND
              klmnn = KlmnnND(obj.X,obj.y);
              % It starts experiments
              experiments = klmnn.runExperiments(...
                methods{i},...
                obj.num_experiments,...                
                obj.num_untrained_classes,...
                obj.training_ratio,...
                obj.random_select_classes,...
                obj.plot_metric);              
              % Save experiments
              save(file,'-struct','experiments');
            catch
              fprintf('\n klmnn experiment error!!! \n');
            end
          case 'knfst'
            try
              % Output file
              file = strcat(obj.out_dir,'/knfst_experiments.mat');
              if exist(file,'file')
                fprintf('Experiment for KNFST already exists!\n');
                continue;
              end               
              % Cria um objeto da classe KnfstND
              knfst = KnfstND(obj.X,obj.y);
              % Inicia experimentos
              experiments = knfst.runExperiments(...
                methods{i},...
                obj.num_experiments,...                
                obj.num_untrained_classes,...
                obj.training_ratio,...
                obj.random_select_classes,...
                obj.plot_metric);
              % Save experiments
              save(file,'-struct','experiments');
            catch
              fprintf('\n knfst experiment error!!! \n');
            end
          case 'one_svm'
            try
              % Output file
              file = strcat(obj.out_dir,'/one_svm_experiments.mat');
              if exist(file,'file')
                fprintf('Experiment for ONE SVM already exists!\n');
                continue;
              end              
              % Cria um objeto da classe SvmND
              one_svm = SvmND(obj.X,obj.y);
              % Inicia experimentos
              experiments = one_svm.runOneSVMExperiments(...
                methods{i},...
                obj.num_experiments,...                
                obj.num_untrained_classes,...
                obj.training_ratio,...
                obj.random_select_classes,...
                obj.plot_metric);
              % Save experiments
              save(file,'-struct','experiments');
            catch
              fprintf('\n one svm experiment error!!! \n');
            end
          case 'multi_svm'
            try
              % Output file
              file = strcat(obj.out_dir,'/multi_svm_experiments.mat');
              if exist(file,'file')
                fprintf('Experiment for MULTI SVM already exists!\n');
                continue;
              end                
              % Cria um objeto da classe SvmND
              multi_svm = SvmND(obj.X,obj.y);
              % Inicia experimentos
              experiments = multi_svm.runMultiSVMExperiments(...
                methods{i},...
                obj.num_experiments,...                
                obj.num_untrained_classes,...
                obj.training_ratio,...
                obj.random_select_classes,...
                obj.plot_metric);
              % Save experiments
              save(file,'-struct','experiments');
            catch
              fprintf('\n multi svm experiment error!!! \n');
            end
          case 'kpca'
            try
              % Output file
              file = strcat(obj.out_dir,'/kpca_experiments.mat');
              if exist(file,'file')
                fprintf('Experiment for KPCA already exists!\n');
                continue;
              end
              % Cria um objeto da classe KpcaND
              kpca = KpcaND(obj.X,obj.y);
              % Inicia experimentos
              experiments = kpca.runExperiments(...
                methods{i},...
                obj.num_experiments,...                
                obj.num_untrained_classes,...
                obj.training_ratio,...
                obj.random_select_classes,...
                obj.plot_metric);
              % Save experiments
              save(file,'-struct','experiments');
            catch
              fprintf('\n kpca experiment error!!! \n');
            end
        end
      end
    end
    
    function runExperimentsForKnnMethods(obj,methods,num_knn_args)
      %-----------------------------------------------------------------------------------
      % This method runs validation experiments and hyperparameter search for the 
      % knn methods.
      %
      % Input args
      %   methods: a cell array of structs with hyperparameters for knn-based novelty 
      %     detection methods.
      %   num_knn_args: number of knn hyperparameters to be tested.
      % ----------------------------------------------------------------------------------            
      if nargin<3
        num_knn_args = 5;
      end
      for K = 1:num_knn_args
        for kappa = 1:K
          fprintf('\nK = %d \tkappa = %d\n',K,kappa);
          % Its sets K and kappa hyperparameters for knn methods.
          for i=1:numel(methods)
            methods{i}.knn_arg = K;
            methods{i}.kappa_threshold = kappa;
          end
          obj.runExperiments(methods);
        end
      end
    end
    
    function runEvaluationTests(obj,xtest,ytest,methods,model_dir)
      % ----------------------------------------------------------------------------------
      % This method is used to evaluate the predictions with multi-class novelty 
      % detection on test sets for the chosen algorithm.
      %
      % Input args
      %   method: a list of strings corresponding to the novelty detection methods used
      %     It can be 'knn','lmnn','klmnn','knfst','one_svm','multi_svm' or 'kpca'.
      % ----------------------------------------------------------------------------------
      xtrain = obj.X(obj.y~=-1,:);
      ytrain = obj.y(obj.y~=-1);      
      for i=1:numel(methods)
        switch methods{i}.name
          case 'knn'
            try
              fprintf('\n-> KNN Novelty Detection \n');
              % Choose the best knn experiment
              experiment = load(strcat(model_dir,'/','best_knn_experiment.mat'));
              knn_dir = strcat(model_dir,'/K=',int2str(experiment.best_K(1)),...
                ' kappa=',int2str(experiment.best_kappa(1)));              
              % Load the model
              best_experiment = load(strcat(knn_dir,'/knn_experiments.mat'));
              model = best_experiment.model;
              % File report
              file_report = strcat(knn_dir,'/knn_evaluation_tests.mat');
              if exist(file_report ,'file')
                fprintf('Evaluation report for KNN (K = %d kappa = %d) already exists!\n',...
                  model.knn_arg,model.kappa_threshold)
                continue;
              end               
              % Run evaluation
              t0_knn = tic;
              knn = KnnND(xtrain,ytrain,model.knn_arg,model.kappa_threshold);
              [knn_evaluations.results,knn_evaluations.evaluations] = ...
                knn.evaluateTests(xtrain,ytrain,xtest,ytest,model);
              knn_evaluations.model = model;
              knn_evaluations.evaluation_time = toc(t0_knn);
              % Save predictions
              save(file_report,'-struct','knn_evaluations');
            catch
              fprintf('\n--> knn evaluation error!!! \n');
            end
          case 'lmnn'
            try
              fprintf('\n-> LMNN Novelty Detection \n');
              % Choose the best knn experiment
              experiment = load(strcat(model_dir,'/','best_lmnn_experiment.mat'));
              knn_dir = strcat(model_dir,'/K=',int2str(experiment.best_K(1)),...
                ' kappa=',int2str(experiment.best_kappa(1)));
              % Load the model
              best_experiment = load(strcat(knn_dir,'/lmnn_experiments.mat'));
              model = best_experiment.model;
              % File report
              file_report = strcat(knn_dir,'/lmnn_evaluation_tests.mat');
              if exist(file_report ,'file')
                fprintf('Evaluation report for LMNN (K = %d kappa = %d) already exists!\n',...
                  model.knn_arg,model.kappa_threshold)
                continue;
              end                        
              % Run evaluation
              t0_lmnn = tic;
              lmnn = LmnnND(xtrain,ytrain,model.knn_arg,model.kappa_threshold);
              [lmnn_evaluations.results,lmnn_evaluations.evaluations] = ...
                lmnn.evaluateTests(xtrain,ytrain,xtest,ytest,model);
              lmnn_evaluations.model = model;
              lmnn_evaluations.evaluation_time = toc(t0_lmnn);
              % Save predictions
              save(file_report,'-struct','lmnn_evaluations');
            catch
              fprintf('\n--> lmnn evaluation error!!! \n');
            end
          case 'klmnn'
            try
              fprintf('\n-> KLMNN Novelty Detection \n');
              % Choose the best knn experiment
              experiment = load(strcat(model_dir,'/','best_klmnn_experiment.mat'));
              knn_dir = strcat(model_dir,'/K=',int2str(experiment.best_K(1)),...
                ' kappa=',int2str(experiment.best_kappa(1)));                            
              % Load the model
              best_experiment = load(strcat(knn_dir,'/klmnn_experiments.mat'));
              model = best_experiment.model;
              % File report
              file_report = strcat(knn_dir,'/klmnn_evaluation_tests.mat');
              if exist(file_report ,'file')
                fprintf('Evaluation report for KLMNN (K = %d kappa = %d) already exists!\n',...
                  model.knn_arg,model.kappa_threshold)
                continue;
              end                                      
              % Run evaluation
              t0_klmnn = tic;
              klmnn = KlmnnND(xtrain,ytrain,model.knn_arg,model.kappa_threshold);
              [klmnn_evaluations.results,klmnn_evaluations.evaluations] = ...
                klmnn.evaluateTests(xtrain,ytrain,xtest,ytest,model);
              klmnn_evaluations.model = model;
              klmnn_evaluations.evaluation_time = toc(t0_klmnn);
              % Save predictions
              save(file_report,'-struct','klmnn_evaluations');
            catch
              fprintf('\n--> klmnn evaluation error!!! \n');
            end
          case 'knfst'
            try
              fprintf('\n-> KNFST Novelty Detection \n');
              % Load the model
              experiments = load(strcat(model_dir,'/knfst_experiments.mat'));
              model = experiments.model;              
              % File report
              file_report = strcat(model_dir,'/knfst_evaluation_tests.mat');                            
              if exist(file_report ,'file')
                fprintf('Evaluation report for KNFST already exists!\n');
                continue;
              end                      
              % Run evaluation
              t0_knfst = tic;
              knfst = KnfstND(xtrain,ytrain);
              [knfst_evaluations.results,knfst_evaluations.evaluations] = ...
                knfst.evaluateTests(xtrain,ytrain,xtest,ytest,model);
              knfst_evaluations.model = model;
              knfst_evaluations.evaluation_time = toc(t0_knfst);
              % Save predictions
              save(file_report,'-struct','knfst_evaluations');
            catch
              fprintf('\n--> knfst evaluation error!!! \n');
            end
          case 'one_svm'
            try
              fprintf('\n-> One SVM Novelty Detection \n');
              % Load the model
              experiments = load(strcat(model_dir,'/one_svm_experiments.mat'));
              model = experiments.model;              
              % File report
              file_report = strcat(model_dir,'/one_svm_evaluation_tests.mat');                            
              if exist(file_report ,'file')
                fprintf('Evaluation report for ONE SVM already exists!\n');
                continue;
              end                      
              % Run evaluation
              t0_one_svm = tic;
              one_svm = SvmND(xtrain,ytrain);
              [one_svm_evaluations.results,one_svm_evaluations.evaluations] = ...
                one_svm.evaluateOneSVMTests(xtrain,ytrain,xtest,ytest,model);
              one_svm_evaluations.model = model;
              one_svm_evaluations.evaluation_time = toc(t0_one_svm);
              % Save predictions
              save(file_report,'-struct','one_svm_evaluations');
            catch
              fprintf('\n--> one svm evaluation error!!! \n');
            end
          case 'multi_svm'
            try
              fprintf('\n-> Multi SVM Novelty Detection \n');
              % Load the model
              experiments = load(strcat(model_dir,'/multi_svm_experiments.mat'));
              model = experiments.model;              
              % File report
              file_report = strcat(model_dir,'/multi_svm_evaluation_tests.mat');                            
              if exist(file_report ,'file')
                fprintf('Evaluation report for MULTI SVM already exists!\n');
                continue;
              end  
              % Run evaluation
              t0_multi_svm = tic;
              multi_svm = SvmND(xtrain,ytrain);
              [multi_svm_evaluations.results,multi_svm_evaluations.evaluations] = ...
                multi_svm.evaluateMultiSVMTests(xtrain,ytrain,xtest,ytest,model);
              multi_svm_evaluations.model = model;
              multi_svm_evaluations.evaluation_time = toc(t0_multi_svm);
              % Save predictions
              save(file_report,'-struct','multi_svm_evaluations');
            catch
              fprintf('\n--> multi svm evaluation error!!! \n');
            end
          case 'kpca'
            try
              fprintf('\n-> KPCA Novelty Detection \n');
              % Load the model
              experiments = load(strcat(model_dir,'/kpca_experiments.mat'));
              model = experiments.model;              
              % File report
              file_report = strcat(model_dir,'/kpca_evaluation_tests.mat');                            
              if exist(file_report ,'file')
                fprintf('Evaluation report for KPCA already exists!\n');
                continue;
              end
              % Run evaluation
              t0_kpca = tic;
              kpca = KpcaND(xtrain,ytrain);
              [kpca_evaluations.results,kpca_evaluations.evaluations] = ...
                kpca.evaluateTests(xtrain,xtest,ytest,model);
              kpca_evaluations.model = model;
              kpca_evaluations.evaluation_time = toc(t0_kpca);
              % Save predictions
              save(file_report,'-struct','kpca_evaluations');
            catch
              fprintf('\n--> kpca nov evaluation error!!! \n');
            end
        end
      end
    end
    
    function runPredictions(obj,xtest,methods,model_dir)
      % ----------------------------------------------------------------------------------
      % This method is used to run the prediction with multi-class novelty detection 
      % in a test set for the chosen methods.
      %
      % Input args
      %   xtest: test data [num_test x dimensions].
      %   methods:
      %   model_dir: model directory.
      % ----------------------------------------------------------------------------------          
      xtrain = obj.X(obj.y~=-1,:);
      ytrain = obj.y(obj.y~=-1);
      for i=1:numel(methods)
        switch methods{i}.name
          case 'knn'
            fprintf('\n-> KNN Novelty Detection \n');
            % Choose the best knn experiment
            experiment = load(strcat(model_dir,'/','best_knn_experiment.mat'));
            knn_dir = strcat(model_dir,'/K=',int2str(experiment.best_K(1)),...
              ' kappa=',int2str(experiment.best_kappa(1)));
            % Load the model
            best_experiment = load(strcat(knn_dir,'/knn_experiments.mat'));            
            model = best_experiment.model;
            % Run predictions
            t0_knn = tic;
            knn = KnnND(xtrain,ytrain,model.knn_arg,model.kappa_threshold);
            predictions = knn.predict(xtrain,ytrain,xtest,model.decision_threshold);
            prediction_time = toc(t0_knn);
            fprintf('\n--> Ok [%.4f s]\n',prediction_time);
            % Save predictions
            save(strcat(knn_dir,'/knn_predictions.mat'),...
              'prediction_time','predictions','xtest');
            % Plot the decision boundaries
            figure;
            Util.plotDecisionBoundary(xtest,predictions);
            Util.plotClassesAux(obj.X,obj.y);
          case 'lmnn'
            fprintf('\n-> LMNN Novelty Detection\n');
            % Choose the best lmnn experiment
            experiment = load(strcat(model_dir,'/','best_lmnn_experiment.mat'));
            knn_dir = strcat(model_dir,'/K=',int2str(experiment.best_K(1)),...
              ' kappa=',int2str(experiment.best_kappa(1)));
            % Load the model
            best_experiment = load(strcat(knn_dir,'/lmnn_experiments.mat'));            
            model = best_experiment.model;            
            % Run predictions
            t0_lmnn = tic;
            lmnn = LmnnND(xtrain,ytrain,model.knn_arg,model.kappa_threshold);
            predictions = lmnn.predict(xtrain,ytrain,xtest,model.decision_threshold);
            prediction_time = toc(t0_lmnn);
            fprintf('\n--> Ok [%.4f s]\n',prediction_time);
            % Save predictions 
            save(strcat(knn_dir,'/lmnn_predictions.mat'),...
              'prediction_time','predictions','xtest');
            % Plot the decision boundaries
            figure;
            Util.plotDecisionBoundary(xtest,predictions);
            Util.plotClassesAux(obj.X,obj.y);
          case 'klmnn'
            fprintf('\n-> KLMNN Novelty Detection\n');
            % Choose the best lmnn experiment
            experiment = load(strcat(model_dir,'/','best_klmnn_experiment.mat'));
            knn_dir = strcat(model_dir,'/K=',int2str(experiment.best_K(1)),...
              ' kappa=',int2str(experiment.best_kappa(1)));
            % Load the model
            best_experiment = load(strcat(knn_dir,'/klmnn_experiments.mat'));            
            model = best_experiment.model;      
            % Run predictions
            t0_klmnn = tic;
            klmnn = KlmnnND(xtrain,ytrain,model.knn_arg,model.kappa_threshold);
            klmnn.kernel_type = model.kernel_type;
            klmnn.reduction_ratio = model.reduction_ratio;
            predictions = klmnn.predict(xtrain,ytrain,xtest,model.kernel,model.decision_threshold);
            prediction_time = toc(t0_klmnn);
            fprintf('-> done! [%.4f s]\n',prediction_time);
            % Save predictions
            save(strcat(knn_dir,'/klmnn_predictions.mat'),...
              'prediction_time','predictions','xtest');
            % Plot the decision boundaries
            figure;
            Util.plotDecisionBoundary(xtest,predictions);
            Util.plotClassesAux(obj.X,obj.y);
          case 'knfst'
            fprintf('\n-> KNFST Novelty Detection\n');
            % Load the model
            experiments = load(strcat(model_dir,'/knfst_experiments.mat'));
            model = experiments.model;
            % Run predictions
            t0_knfst = tic;
            knfst = KnfstND(xtrain,ytrain);
            knfst.kernel_type = model.kernel_type;
            predictions = knfst.predict(xtrain,ytrain,xtest,...
              model.kernel,model.decision_threshold);
            prediction_time = toc(t0_knfst);
            fprintf('\n--> Ok [%.4f s]\n',prediction_time);
            % Save predictions
            save(strcat(model_dir,'/knfst_predictions.mat'),...
              'prediction_time','predictions','xtest');
            % Plot the decision boundaries
            figure;
            Util.plotDecisionBoundary(xtest,predictions);
            Util.plotClassesAux(obj.X,obj.y);
          case 'one_svm'
            fprintf('\n-> One SVM Novelty Detection\n');
            % Load the model
            experiments = load(strcat(model_dir,'/one_svm_experiments.mat'));
            model = experiments.model;
            % Run predictions
            t0_svm = tic;
            one_svm = SvmND(xtrain,ytrain,obj.num_classes);
            one_svm.kernel_type = model.kernel_type;
            predictions = one_svm.predictOneSVM(xtrain,ytrain,xtest,model.kernel);
            prediction_time = toc(t0_svm);
            fprintf('\n--> Ok [%.4f s]\n',prediction_time);
            % Save predictions
            save(strcat(model_dir,'/one_svm_predictions.mat'),'prediction_time',...
              'predictions','xtest');
            % Plot the decision boundaries
            figure;
            Util.plotDecisionBoundary(xtest,predictions);
            Util.plotClassesAux(obj.X,obj.y);
          case 'multi_svm'
            fprintf('\n-> Multi SVM Novelty Detection\n');
            % Load the model
            experiments = load(strcat(model_dir,'/multi_svm_experiments.mat'));
            model = experiments.model;
            % Run predictions
            t0_svm = tic;
            multi_svm = SvmND(xtrain,ytrain,obj.num_classes);
            multi_svm.kernel_type = model.kernel_type;
            predictions = multi_svm.predictMultiSVM(xtrain,ytrain,xtest,...
              model.kernel,model.decision_threshold);
            prediction_time = toc(t0_svm);
            fprintf('\n--> Ok [%.4f s]\n',prediction_time);
            % Save predictions
            save(strcat(model_dir,'/multi_svm_predictions.mat'),'prediction_time',...
              'predictions','xtest');
            % Plot the decision boundaries
            figure;
            Util.plotDecisionBoundary(xtest,predictions);
            Util.plotClassesAux(obj.X,obj.y);
          case 'kpca'
            fprintf('\n-> KPCA Novelty Detection\n');
            % Load the model
            experiments = load(strcat(model_dir,'/kpca_experiments.mat'));
            model = experiments.model;
            % Run predictions
            t0_kpca = tic;
            kpca = KpcaND(xtrain,ytrain,obj.num_classes);
            kpca.kernel_type = model.kernel_type;
            predictions = kpca.predict(xtrain,xtest,...
              model.kernel,model.decision_threshold);
            prediction_time = toc(t0_kpca);
            fprintf('\n--> Ok [%.4f s]\n',prediction_time);
            % Save predictions
            save(strcat(model_dir,'/kpca_predictions.mat'),...
              'prediction_time','predictions','xtest');
            % Plot the decision boundaries
            figure;
            Util.plotDecisionBoundary(xtest,predictions);
            Util.plotClassesAux(obj.X,obj.y);
        end
      end
    end
    
    function runPredictionsForKnnMethods(obj,methods,xtrain,ytrain,xtest,ytest,model_dir)
      % ----------------------------------------------------------------------------------
      % This method is used to run the prediction with multi-class novelty detection 
      % in a test set for the knn methods.
      %
      % Input args
      %   xtest: test data [num_test x dimensions].
      %   model_dir: model directory.
      % ----------------------------------------------------------------------------------          
      % Testes
      for K = 1:5
        for kappa = K:-1:K-3
          if kappa >= 1
            fprintf('\nK = %d \tkappa = %d\n',K,kappa);
            Manager.runPredict(obj,xtrain,ytrain,xtest,ytest,methods,...
              model_dir,obj.num_experiments,K,kappa);
          end
        end
      end
    end
    
    function reportExperiments(obj,out_dir,methods)
      % ----------------------------------------------------------------------------------
      % This method is used to load and process results of novelty detection experiments 
      % for the chosen algorithm.
      %
      % Inputa args
      %   out_dir: experiment diretory.
      %   methods: methods to be processed.
      % ----------------------------------------------------------------------------------      
      fprintf('\nProcessing results... ');      
      num_methods = numel(methods);
      TPR = zeros(1,num_methods);
      TNR = zeros(1,num_methods);
      FPR = zeros(1,num_methods);
      FNR = zeros(1,num_methods);
      AFR = zeros(1,num_methods);
      F1 = zeros(1,num_methods);
      MCC = zeros(1,num_methods);
      for k=1:num_methods 
        switch methods{k}.name
          case 'knn'
            try              
              report = load(strcat(out_dir,'/','best_knn_experiment.mat'));
              knn_dir = strcat(out_dir,'/K=',int2str(report.best_K(1)),...
                ' kappa=',int2str(report.best_kappa(1)));
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
              report = load(strcat(out_dir,'/','best_lmnn_experiment.mat'));
              knn_dir = strcat(out_dir,'/K=',int2str(report.best_K(1)),...
                ' kappa=',int2str(report.best_kappa(1)));
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
              report = load(strcat(out_dir,'/','best_klmnn_experiment.mat'));
              knn_dir = strcat(out_dir,'/K=',int2str(report.best_K(1)),...
                ' kappa=',int2str(report.best_kappa(1)));
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
              experiment = load(strcat(out_dir,'/knfst_experiments.mat'));
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
              experiment = load(strcat(out_dir,'/one_svm_experiments.mat'));
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
              experiment = load(strcat(out_dir,'/multi_svm_experiments.mat'));
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
              experiment = load(strcat(out_dir,'/kpca_experiments.mat'));
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
      
      save(strcat(out_dir,'/report_results.mat'),'TPR','TNR',...
        'FPR','FNR','AFR','F1','MCC','REPORT');
      writetable(REPORT,strcat(out_dir,'/report_results.csv'),...
        'WriteRowNames',true,'Delimiter',';');
      fprintf('done!\n');
    end
    
    function reports = reportExperimentsForKnnMethods(obj,out_dir,num_knn_args)
      % ----------------------------------------------------------------------------------
      % This method is used to load and process results of novelty detection experiments 
      % for knn methods.
      %
      % Input args
      %   out_dir: experiment directory.
      %   num_knn_args: number of knn hyperparameters tested.
      % Output args
      %   reports: reports for knn, lmnn and klmnn-based novelty detection methods.
      % ----------------------------------------------------------------------------------
      report_file = strcat([out_dir,'/','report_knn_methods.mat']);
      if ~exist(report_file,'file')
        if nargin<3
          num_knn_args = 5;
        end
        fprintf('\nLoading experiment results... ');
        % KNN
        KNN.TPR = nan*zeros(num_knn_args,num_knn_args); 
        KNN.TNR = nan*zeros(num_knn_args,num_knn_args);
        KNN.FPR = nan*zeros(num_knn_args,num_knn_args); 
        KNN.FNR = nan*zeros(num_knn_args,num_knn_args);
        KNN.AFR = nan*zeros(num_knn_args,num_knn_args); 
        KNN.F1 = nan*zeros(num_knn_args,num_knn_args);
        KNN.MCC = nan*zeros(num_knn_args,num_knn_args);
        % LMNN
        LMNN.TPR = nan*zeros(num_knn_args,num_knn_args); 
        LMNN.TNR = nan*zeros(num_knn_args,num_knn_args);
        LMNN.FPR = nan*zeros(num_knn_args,num_knn_args); 
        LMNN.FNR = nan*zeros(num_knn_args,num_knn_args);
        LMNN.AFR = nan*zeros(num_knn_args,num_knn_args); 
        LMNN.F1 = nan*zeros(num_knn_args,num_knn_args);
        LMNN.MCC = nan*zeros(num_knn_args,num_knn_args);
        % KLMNN
        KLMNN.TPR = nan*zeros(num_knn_args,num_knn_args); 
        KLMNN.TNR = nan*zeros(num_knn_args,num_knn_args);
        KLMNN.FPR = nan*zeros(num_knn_args,num_knn_args); 
        KLMNN.FNR = nan*zeros(num_knn_args,num_knn_args);
        KLMNN.AFR = nan*zeros(num_knn_args,num_knn_args); 
        KLMNN.F1 = nan*zeros(num_knn_args,num_knn_args);
        KLMNN.MCC = nan*zeros(num_knn_args,num_knn_args);

        % Testes
        for K = 1:num_knn_args
          for kappa = 1:K
            knn_dir = strcat(out_dir,'/K=',int2str(K),' kappa=',int2str(kappa));
            fprintf('\nK = %d \tkappa = %d\n',K,kappa);
            % KNN
            try
              file = strcat(knn_dir,'/knn_experiments.mat');
              experiment = load(file);
              KNN.TPR(K,kappa) = experiment.tpr_score;
              KNN.TNR(K,kappa) = experiment.tnr_score;
              KNN.FPR(K,kappa) = experiment.fpr_score;
              KNN.FNR(K,kappa) = experiment.fnr_score;
              KNN.AFR(K,kappa) = experiment.afr_score;
              KNN.F1(K,kappa) = experiment.f1_score;
              KNN.MCC(K,kappa) = experiment.mcc_score;
            catch
              fprintf('\n--> error knn results!\n');
            end
            % LMNN
            try
              file = strcat(knn_dir,'/lmnn_experiments.mat');
              experiment = load(file);
              LMNN.TPR(K,kappa) = experiment.tpr_score;
              LMNN.TNR(K,kappa) = experiment.tnr_score;
              LMNN.FPR(K,kappa) = experiment.fpr_score;
              LMNN.FNR(K,kappa) = experiment.fnr_score;
              LMNN.AFR(K,kappa) = experiment.afr_score;
              LMNN.F1(K,kappa) = experiment.f1_score;
              LMNN.MCC(K,kappa) = experiment.mcc_score;
            catch
              fprintf('\n--> error lmnn results!\n');
            end
            % KLMNN
            try
              file = strcat(knn_dir,'/klmnn_experiments.mat');
              experiment = load(file);
              KLMNN.TPR(K,kappa) = experiment.tpr_score;
              KLMNN.TNR(K,kappa) = experiment.tnr_score;
              KLMNN.FPR(K,kappa) = experiment.fpr_score;
              KLMNN.FNR(K,kappa) = experiment.fnr_score;
              KLMNN.AFR(K,kappa) = experiment.afr_score;
              KLMNN.F1(K,kappa) = experiment.f1_score;
              KLMNN.MCC(K,kappa) = experiment.mcc_score;
            catch
              fprintf('\n--> error klmnn results!\n');
            end
          end
        end
        % Cria as tabelas
        kappa = struct();
        for k=1:num_knn_args
          K_names{k,1} = sprintf('K = %d',k);
          kappa_names{k} = sprintf('kappa%d',k);
          kappa.(kappa_names{k}) = sprintf('kappa = %d',k);
        end        
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

        save(strcat(out_dir,'/report_knn_methods.mat'),'kappa','KNN','LMNN','KLMNN');
        fprintf('done!\n');     
      end
      report = load(report_file);
      % KNN
      KNN_MCC = table2array(report.KNN.MCC);
      best_mcc = max(max(KNN_MCC));
      [K,kappa] = find(KNN_MCC==best_mcc);
      
      knn_report.name = 'knn';
      knn_report.best_mcc = best_mcc;
      knn_report.best_K = K;
      knn_report.best_kappa = kappa;
      
      % LMNN
      LMNN_MCC = table2array(report.LMNN.MCC);
      best_mcc = max(max(LMNN_MCC));
      [K,kappa] = find(LMNN_MCC==best_mcc);
      
      lmnn_report.name = 'lmnn';
      lmnn_report.best_mcc = best_mcc;
      lmnn_report.best_K = K;
      lmnn_report.best_kappa = kappa;      
      
      % KLMNN
      KLMNN_MCC = table2array(report.KLMNN.MCC);
      best_mcc = max(max(KLMNN_MCC));
      [K,kappa] = find(KLMNN_MCC==best_mcc);
      
      klmnn_report.name = 'klmnn';
      klmnn_report.best_mcc = best_mcc;
      klmnn_report.best_K = K;
      klmnn_report.best_kappa = kappa;            
      
      fprintf('------------------------ KNN ------------------------\n');      
      fprintf('best K:');
      fprintf(' %d',knn_report.best_K);      
      fprintf('\nbest kappa:');
      fprintf(' %d',knn_report.best_kappa);
      fprintf('\nbest MCC: %4f\n',knn_report.best_mcc);
      fprintf('Full MCC table:\n');
      disp(report.KNN.MCC);
      
      fprintf('------------------------ LMNN ------------------------\n');      
      fprintf('best K:');
      fprintf(' %d',lmnn_report.best_K);      
      fprintf('\nbest kappa:');
      fprintf(' %d',lmnn_report.best_kappa);
      fprintf('\nbest MCC: %4f\n',lmnn_report.best_mcc);
      fprintf('Full MCC table:\n');
      disp(report.LMNN.MCC);
      
      fprintf('------------------------ KLMNN ------------------------\n');      
      fprintf('best K:');
      fprintf(' %d',klmnn_report.best_K);      
      fprintf('\nbest kappa:');
      fprintf(' %d',klmnn_report.best_kappa);
      fprintf('\nbest MCC: %4f\n',klmnn_report.best_mcc);
      fprintf('Full MCC table:\n');
      disp(report.KLMNN.MCC);            
      
      save(strcat([out_dir,'/','best_knn_experiment.mat']),'-struct','knn_report');
      save(strcat([out_dir,'/','best_lmnn_experiment.mat']),'-struct','lmnn_report');
      save(strcat([out_dir,'/','best_klmnn_experiment.mat']),'-struct','klmnn_report');
      
      reports = {knn_report,lmnn_report,klmnn_report};
    end
    
    function reportDataVariationExperiment(obj,methods,model_dir,N,DIM)
      % ----------------------------------------------------------------------------------
      % This method is used to load execution times and compute accuracy metrics
      % of test sets for data variation experiments.
      %
      % Input args
      %   methods: a cell array of structs with method configurations.
      %   model_dir: model directory.
      %   N: number of samples in the training set.
      %   DIM: spatial dimension of samples in the training set.
      % ----------------------------------------------------------------------------------            
      exp_file = strcat(model_dir,'/report_experiment_1.mat');
      if ~exist(exp_file,'file')
        fprintf('Processing results for data variation experiment... ');
        exp.MCC = zeros(numel(methods),numel(N));
        exp.F1 = zeros(numel(methods),numel(N));
        exp.val_time = zeros(numel(methods),numel(N));
        exp.mean_test_time = zeros(numel(methods),numel(N));          
        % Variation in the number of training samples
        for j=1:numel(N)
          exp_dir = strcat(model_dir,'/N=',int2str(N(j)),' DIM=',int2str(DIM));
          for i=1:numel(methods)
            switch methods{i}.name
              case 'knn'
                try
                  % Choose the best knn experiment
                  experiment = load(strcat(exp_dir,'/','best_knn_experiment.mat'));
                  knn_dir = strcat(exp_dir,'/K=',int2str(experiment.best_K(1)),...
                    ' kappa=',int2str(experiment.best_kappa(1)));              
                  % Load the experiment
                  best_experiment = load(strcat(knn_dir,'/knn_experiments.mat'));              
                  % Load evaluations                                   
                  file_test = strcat(knn_dir,'/knn_evaluation_tests.mat');
                  load(file_test,'results','evaluation_time');
                  exp.MCC(i,j) = mean(results.MCC);
                  exp.F1(i,j) = mean(results.F1);
                  exp.mean_test_time(i,j) = evaluation_time/size(results,1);
                  exp.val_time(i,j) = best_experiment.total_time;                  
                catch
                  fprintf('\n--> error processing  knn!\n');
                end
              case 'lmnn'
                try
                  % Choose the best knn experiment
                  experiment = load(strcat(exp_dir,'/','best_lmnn_experiment.mat'));
                  knn_dir = strcat(exp_dir,'/K=',int2str(experiment.best_K(1)),...
                    ' kappa=',int2str(experiment.best_kappa(1)));              
                  % Load the experiment
                  best_experiment = load(strcat(knn_dir,'/lmnn_experiments.mat'));                  
                  % Load evaluations                  
                  file_test = strcat(knn_dir,'/lmnn_evaluation_tests.mat');
                  load(file_test,'results','evaluation_time');
                  exp.MCC(i,j) = mean(results.MCC);
                  exp.F1(i,j) = mean(results.F1);
                  exp.mean_test_time(i,j) = evaluation_time/size(results,1);
                  exp.val_time(i,j) = best_experiment.total_time;
                catch
                  fprintf('\n--> error processing  lmnn!\n');
                end
              case 'klmnn'
                try
                  % Choose the best knn experiment
                  experiment = load(strcat(exp_dir,'/','best_klmnn_experiment.mat'));
                  knn_dir = strcat(exp_dir,'/K=',int2str(experiment.best_K(1)),...
                    ' kappa=',int2str(experiment.best_kappa(1)));              
                  % Load the experiment
                  best_experiment = load(strcat(knn_dir,'/klmnn_experiments.mat'));                    
                  % Load evaluations                  
                  file_test = strcat(knn_dir,'/klmnn_evaluation_tests.mat');
                  load(file_test,'results','evaluation_time');
                  exp.MCC(i,j) = mean(results.MCC);
                  exp.F1(i,j) = mean(results.F1);
                  exp.mean_test_time(i,j) = evaluation_time/size(results,1);
                  exp.val_time(i,j) = best_experiment.total_time;    
                catch
                  fprintf('\n--> error processing  klmnn!\n');
                end
              case 'knfst'
                try
                  % Load the experiment
                  experiment = load(strcat(exp_dir,'/knfst_experiments.mat'));                    
                  % Load evaluations
                  file_test = strcat(exp_dir,'/knfst_evaluation_tests.mat');
                  load(file_test,'results','evaluation_time');
                  exp.MCC(i,j) = mean(results.MCC);
                  exp.F1(i,j) = mean(results.F1);
                  exp.mean_test_time(i,j) = evaluation_time/size(results,1);
                  exp.val_time(i,j) = experiment.total_time;    
                catch
                  fprintf('\n--> error processing  knfst!\n');
                end
              case 'one_svm'
                try
                  % Load the experiment
                  experiment = load(strcat(exp_dir,'/one_svm_experiments.mat'));                    
                  % Load evaluations                                   
                  file_test = strcat(exp_dir,'/one_svm_evaluation_tests.mat');
                  load(file_test,'results','evaluation_time');
                  exp.MCC(i,j) = mean(results.MCC);
                  exp.F1(i,j) = mean(results.F1);
                  exp.mean_test_time(i,j) = evaluation_time/size(results,1);
                  exp.val_time(i,j) = experiment.total_time;    
                catch
                  fprintf('\n--> error processing  one svm!\n');
                end
              case 'multi_svm'
                try
                  % Load the experiment
                  experiment = load(strcat(exp_dir,'/multi_svm_experiments.mat'));                    
                  % Load evaluations       
                  file_test = strcat(exp_dir,'/multi_svm_evaluation_tests.mat');
                  load(file_test,'results','evaluation_time');
                  exp.MCC(i,j) = mean(results.MCC);
                  exp.F1(i,j) = mean(results.F1);
                  exp.mean_test_time(i,j) = evaluation_time/size(results,1);
                  exp.val_time(i,j) = experiment.total_time;  
                catch
                  fprintf('\n--> error processing  multi svm!\n');
                end
              case 'kpca'
                try
                  % Load the experiment
                  experiment = load(strcat(exp_dir,'/multi_svm_experiments.mat'));                    
                  % Load evaluations    
                  file_test = strcat(exp_dir,'/kpca_evaluation_tests.mat');
                  load(file_test,'results','evaluation_time');
                  exp.MCC(i,j) = mean(results.MCC);
                  exp.F1(i,j) = mean(results.F1);
                  exp.mean_test_time(i,j) = evaluation_time/size(results,1);
                  exp.val_time(i,j) = experiment.total_time;  
                catch
                  fprintf('\n--> error processing  kpca!\n');
                end
            end
          end
        end
        var_names = split(sprintf('N%d,',N),',');
        var_names = var_names(1:end-1);
        row_names = {'KNN','LMNN','KLMNN','KNFST','ONE_SVM','MULTI_SVM','KPCA_NOV'};
        
        exp.MCC = array2table(exp.MCC,'VariableNames',var_names,'RowNames',row_names);
        exp.F1 = array2table(exp.F1,'VariableNames',var_names,'RowNames',row_names);
        exp.val_time = array2table(exp.val_time,...
          'VariableNames',var_names,'RowNames',row_names);
        exp.mean_test_time = array2table(exp.mean_test_time,...
          'VariableNames',var_names,'RowNames',row_names);
        
        fprintf('done!\n');
      else
        exp = load(exp1_file);
      end
      
      % PLOTS EXPERIMENTO 1
      figure;
      clf('reset');
      %subplot(1,2,1);
      hold on;
      p = plot(N,exp.MCC.Variables','-s','LineWidth',1);
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
      p = plot(N,exp.mean_test_time.Variables','-s','LineWidth',1);
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
      saveas(gcf,strcat(model_dir,'/sim4-exp1-test_time.pdf'));
      saveas(gcf,strcat(model_dir,'/sim4-exp1-test_time.fig'));
      
      figure;
      clf('reset');
      hold on;
      p = plot(N,exp.val_time.Variables'/60,'-s','LineWidth',1);
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
            
      save(strcat(model_dir,'/report_experiment_1.mat'),'-struct','exp');
    end
    
    function reportDimensionVariationExperiment(obj,methods,model_dir,N,DIM)
      % ----------------------------------------------------------------------------------
      % This method is used to load execution times and compute accuracy metrics
      % of test sets for dimension variation experiments.
      %
      % Input args
      %   methods: a cell array of structs with method configurations.
      %   model_dir: model directory.
      %   N: number of samples in the training set.
      %   DIM: spatial dimension of samples in the training set.
      % ----------------------------------------------------------------------------------                  
      exp_file = strcat(model_dir,'/report_experiment_2.mat');
      if ~exist(exp_file,'file')
        fprintf('Processing results for the dimension variation experiment... ');
        exp.MCC = zeros(numel(methods),numel(DIM));
        exp.F1 = zeros(numel(methods),numel(DIM));
        exp.val_time = zeros(numel(methods),numel(DIM));
        exp.mean_test_time = zeros(numel(methods),numel(DIM));        
       % Variation in the number of dimensions
        for j=1:numel(DIM)
          exp_dir = strcat(model_dir,'/N=',int2str(N),' DIM=',int2str(DIM(j)));
          for i=1:numel(methods)
            switch methods{i}.name
              case 'knn'
                try
                  % Choose the best knn experiment
                  experiment = load(strcat(exp_dir,'/','best_knn_experiment.mat'));
                  knn_dir = strcat(exp_dir,'/K=',int2str(experiment.best_K(1)),...
                    ' kappa=',int2str(experiment.best_kappa(1)));              
                  % Load the experiment
                  best_experiment = load(strcat(knn_dir,'/knn_experiments.mat'));              
                  % Load evaluations                                   
                  file_test = strcat(knn_dir,'/knn_evaluation_tests.mat');
                  load(file_test,'results','evaluation_time');
                  exp.MCC(i,j) = mean(results.MCC);
                  exp.F1(i,j) = mean(results.F1);
                  exp.mean_test_time(i,j) = evaluation_time/size(results,1);
                  exp.val_time(i,j) = best_experiment.total_time;                  
                catch
                  fprintf('\n--> error processing  knn!\n');
                end
              case 'lmnn'
                try
                  % Choose the best knn experiment
                  experiment = load(strcat(exp_dir,'/','best_lmnn_experiment.mat'));
                  knn_dir = strcat(exp_dir,'/K=',int2str(experiment.best_K(1)),...
                    ' kappa=',int2str(experiment.best_kappa(1)));              
                  % Load the experiment
                  best_experiment = load(strcat(knn_dir,'/lmnn_experiments.mat'));                  
                  % Load evaluations                  
                  file_test = strcat(knn_dir,'/lmnn_evaluation_tests.mat');
                  load(file_test,'results','evaluation_time');
                  exp.MCC(i,j) = mean(results.MCC);
                  exp.F1(i,j) = mean(results.F1);
                  exp.mean_test_time(i,j) = evaluation_time/size(results,1);
                  exp.val_time(i,j) = best_experiment.total_time;
                catch
                  fprintf('\n--> error processing  lmnn!\n');
                end
              case 'klmnn'
                try
                  % Choose the best knn experiment
                  experiment = load(strcat(exp_dir,'/','best_klmnn_experiment.mat'));
                  knn_dir = strcat(exp_dir,'/K=',int2str(experiment.best_K(1)),...
                    ' kappa=',int2str(experiment.best_kappa(1)));              
                  % Load the experiment
                  best_experiment = load(strcat(knn_dir,'/klmnn_experiments.mat'));                    
                  % Load evaluations                  
                  file_test = strcat(knn_dir,'/klmnn_evaluation_tests.mat');
                  load(file_test,'results','evaluation_time');
                  exp.MCC(i,j) = mean(results.MCC);
                  exp.F1(i,j) = mean(results.F1);
                  exp.mean_test_time(i,j) = evaluation_time/size(results,1);
                  exp.val_time(i,j) = best_experiment.total_time;    
                catch
                  fprintf('\n--> error processing  klmnn!\n');
                end
              case 'knfst'
                try
                  % Load the experiment
                  experiment = load(strcat(exp_dir,'/knfst_experiments.mat'));                    
                  % Load evaluations
                  file_test = strcat(exp_dir,'/knfst_evaluation_tests.mat');
                  load(file_test,'results','evaluation_time');
                  exp.MCC(i,j) = mean(results.MCC);
                  exp.F1(i,j) = mean(results.F1);
                  exp.mean_test_time(i,j) = evaluation_time/size(results,1);
                  exp.val_time(i,j) = experiment.total_time;    
                catch
                  fprintf('\n--> error processing  knfst!\n');
                end
              case 'one_svm'
                try
                  % Load the experiment
                  experiment = load(strcat(exp_dir,'/one_svm_experiments.mat'));                    
                  % Load evaluations                                   
                  file_test = strcat(exp_dir,'/one_svm_evaluation_tests.mat');
                  load(file_test,'results','evaluation_time');
                  exp.MCC(i,j) = mean(results.MCC);
                  exp.F1(i,j) = mean(results.F1);
                  exp.mean_test_time(i,j) = evaluation_time/size(results,1);
                  exp.val_time(i,j) = experiment.total_time;    
                catch
                  fprintf('\n--> error processing  one svm!\n');
                end
              case 'multi_svm'
                try
                  % Load the experiment
                  experiment = load(strcat(exp_dir,'/multi_svm_experiments.mat'));                    
                  % Load evaluations       
                  file_test = strcat(exp_dir,'/multi_svm_evaluation_tests.mat');
                  load(file_test,'results','evaluation_time');
                  exp.MCC(i,j) = mean(results.MCC);
                  exp.F1(i,j) = mean(results.F1);
                  exp.mean_test_time(i,j) = evaluation_time/size(results,1);
                  exp.val_time(i,j) = experiment.total_time;  
                catch
                  fprintf('\n--> error processing  multi svm!\n');
                end
              case 'kpca'
                try
                  % Load the experiment
                  experiment = load(strcat(exp_dir,'/multi_svm_experiments.mat'));                    
                  % Load evaluations    
                  file_test = strcat(exp_dir,'/kpca_evaluation_tests.mat');
                  load(file_test,'results','evaluation_time');
                  exp.MCC(i,j) = mean(results.MCC);
                  exp.F1(i,j) = mean(results.F1);
                  exp.mean_test_time(i,j) = evaluation_time/size(results,1);
                  exp.val_time(i,j) = experiment.total_time;  
                catch
                  fprintf('\n--> error processing  kpca!\n');
                end
            end
          end
        end
        var_names = split(sprintf('D%d,',DIM),',');
        var_names = var_names(1:end-1);
        row_names = {'KNN','LMNN','KLMNN','KNFST','ONE_SVM','MULTI_SVM','KPCA_NOV'};
        
        exp.MCC = array2table(exp.MCC,'VariableNames',var_names,'RowNames',row_names);
        exp.F1 = array2table(exp.F1,'VariableNames',var_names,'RowNames',row_names);
        exp.val_time= array2table(exp.val_time,...
          'VariableNames',var_names,'RowNames',row_names);
        exp.mean_test_time = array2table(exp.mean_test_time,...
          'VariableNames',var_names,'RowNames',row_names);
        
        fprintf('done!\n');
      else
        exp = load(exp_file);
      end
      
      % Plots do experimento 2
      figure;
      %sublplot(1,1,2);
      clf('reset');
      hold on;
      p = plot(DIM,exp.MCC.Variables','-s','LineWidth',1);
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
      saveas(gcf,strcat(model_dir,'/sim4-exp1-mcc.pdf'));
      saveas(gcf,strcat(model_dir,'/sim4-exp1-mcc.fig'));
      
      figure;
      clf('reset');
      %subplot(2,1,2);
      hold on;
      p = plot(DIM,exp.mean_test_time.Variables','-s','LineWidth',1);
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
      saveas(gcf,strcat(model_dir,'/sim4-exp2-test_time.pdf'));
      saveas(gcf,strcat(model_dir,'/sim4-exp2-test_time.fig'));
      
      figure;
      clf('reset');
      hold on;
      p = plot(DIM,exp.val_time.Variables'/60,'-s','LineWidth',1);
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
            
      save(strcat(model_dir,'/report_experiment_2.mat'),'-struct','exp');
    end        
  end
end