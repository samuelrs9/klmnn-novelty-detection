classdef SplitData < handle
  % --------------------------------------------------------------------------------------
  % This class is used to manage the random splitting of dataset into training, validation 
  % and test sets. 
  %
  % Version 2.0, July 2022.
  % By Samuel Silva (samuelrs@usp.br).
  % --------------------------------------------------------------------------------------    
  properties
    X = [];               % samples in dataset
    Y = [];               % sample labels in dataset    
    training_ratio = 0;   % training samples rate
    test_ratio = 0;       % test samples rate
    num_classes = 0;      % number of classes in dataset
    num_samples = 0;      % number of samples in dataset
    istrainedclass = [];  % boolean list, true for trained classes and false for untrained
    classes = [];         % classses indices
    trained = [];         % trained classes list
    untrained = [];       % untrained classes list (used as novelty)
    idx_trained = [];     % Índices de dados de classes "conhecidas"
    idx_untrained = [];   % Índices de dados "desconhecidos (outliers)
    idx_train_val_t = []; % Corresponde a 70% dos dados de classes conhecidas
    idx_test_t = [];      % Corresponde a 30% dos dados de classes conhecidas
    idx_val_u = [];       % corresponde a 50% dos dados de classes desconhecidas
    idx_test_u = [];      % corresponde a 50% dos dados de classes desconhecidas
  end
  
  methods    
    function obj = SplitData(X,Y,training_ratio,num_untrained)
      % ----------------------------------------------------------------------------------
      % Constructor.
      %
      % Args
      %   X: samples [num_samples x dimension].
      %   Y: sample labels [num_samples x 1].
      %   training_ratio: training sample rate.
      %   num_untrained: number of untrained classes, this parameter can
      %     be used to simulate novelty data in the dataset.
      % ----------------------------------------------------------------------------------
      obj.Y = Y;
      obj.X = X;
      obj.training_ratio = training_ratio;
      obj.test_ratio = 0;
      obj.classes = unique(Y);
      obj.num_classes = numel(obj.classes);
      obj.num_samples = numel(Y);
      [obj.trained,obj.untrained,obj.istrainedclass] = obj.selectClasses(num_untrained);
      [obj.idx_trained,obj.idx_untrained] = obj.splitTrainedUntrained(obj.istrainedclass);
      [obj.idx_train_val_t,obj.idx_test_t,obj.idx_val_u,obj.idx_test_u] = obj.idxTrainValTest();
    end
    
    function [trained,untrained,istrainedclass] = selectClasses(obj,num_untrained)
      % ----------------------------------------------------------------------------------    
      % This method selects trained and untrained classes.
      %
      % Input args
      %   num_untrained: if it is an integer, then num_untrained classes will be selected 
      %     randomly and all samples of those classes will be set as novelty. If it is the 
      %     string "auto", the selection will be made automatically, in this case, all 
      %     samples that have the label equal to -1 will be used as novelty.
      %
      % Output args
      %   trained: list of trained classes.
      %   untrained: list of untrained classes.
      %   istrainedclass: list of boolean values indicating whether a class is
      %     trained or untrained.
      % ----------------------------------------------------------------------------------
      if strcmp(num_untrained,"auto")
        untrained = obj.classes(obj.classes==-1);
        trained = obj.classes(obj.classes~=-1);
        istrainedclass = true(obj.num_classes,1);
        istrainedclass(obj.classes==-1) = false;
      else
        idx = randperm(obj.num_classes)';
        istrainedclass = true(obj.num_classes,1);
        istrainedclass(idx(1:num_untrained)) = false;
        untrained = idx(1:num_untrained);
        trained = idx(num_untrained+1:obj.num_classes);
      end
    end
    
    function [idx_trained,idx_untrained] = splitTrainedUntrained(obj,istrainedclass)
      % ----------------------------------------------------------------------------------
      % This method splits classes into trained and untrained classes.
      % ----------------------------------------------------------------------------------
      idx = (1:obj.num_samples)';
      idx_trained = [];
      idx_untrained = [];
      for i=1:obj.num_classes
        if istrainedclass(i)
          idx_trained = [idx_trained;idx(obj.Y==obj.classes(i))];
        else
          idx_untrained = [idx_untrained;idx(obj.Y==obj.classes(i))];
        end
      end
    end
        
    function [id_train_val_t,id_test_t,id_val_u,id_test_u] = idxTrainValTest(obj)
      % ----------------------------------------------------------------------------------
      % Divide os indices dados em treinamento e teste
      % ----------------------------------------------------------------------------------
      % Permuta os dados de classes conhecidas
      num_trained_samples = numel(obj.idx_trained);
      obj.idx_trained = obj.idx_trained(randperm(num_trained_samples));
      
      % Divide os dados de classes conhecidas em treino/validação e teste.
      num_train_val_t = floor((1-obj.test_ratio)*num_trained_samples);
      id_train_val_t = obj.idx_trained(1:num_train_val_t);
      id_test_t = obj.idx_trained(num_train_val_t + 1:num_trained_samples);
      
      % Permuta os dados de classes desconhecidas
      num_untrained_samples = numel(obj.idx_untrained);
      obj.idx_untrained = obj.idx_untrained(randperm(num_untrained_samples));
      
      % Divide os dados de classes desconhecidas em validação e teste.
      num_val_u = floor((1-obj.test_ratio)*num_untrained_samples);
      id_val_u = obj.idx_untrained(1:num_val_u);
      id_test_u = obj.idx_untrained(num_val_u+1:num_untrained_samples);
    end
        
    function [idx_train,idx_val] = idxTrainVal(obj)
      % ----------------------------------------------------------------------------------
      % This method splits a part of the dataset for training and validation.
      %
      % Output args
      %   idx_train: train indices.      
      %   idx_val: validation indices.      
      % ----------------------------------------------------------------------------------
      % Permuta as amostras de treinamento/validação
      num_train_val = numel(obj.idx_train_val_t);
      obj.idx_train_val_t = obj.idx_train_val_t(randperm(num_train_val));
      
      % Divide os dados de treinamento/validação
      num_train = floor(obj.training_ratio*num_train_val);
      num_val_t = num_train_val - num_train;
      
      idx_train = obj.idx_train_val_t(1:num_train);
      idx_val = obj.idx_train_val_t(num_train+1:num_train_val);
      
      % Permuta as amostras de validação desconhecidas
      num_val_u = numel(obj.idx_val_u);
      obj.idx_val_u = obj.idx_val_u(randperm(num_val_u));
      
      % Complementa o conjunto de validação com uma parte dos dados de classes
      % não treinadas destinados para validação
      idx_val = [idx_val;obj.idx_val_u(1:floor(obj.training_ratio*num_val_u))];
    end
    
    function idx_test = idxTest(obj)
      % ----------------------------------------------------------------------------------      
      % This method splits a part of the dataset for test.
      %
      % Output args
      %   idx_test: test indices.
      % ----------------------------------------------------------------------------------
      % Permuta os dados de teste de classes conhecidas
      num_test_t = numel(obj.idx_test_t);
      obj.idx_test_t = obj.idx_test_t(randperm(num_test_t));
      
      % Pega 50% dos dados de teste de classes conhecidas
      idx_test = obj.idx_test_t(1:floor(0.5*num_test_t));
      
      % Permuta os dados de teste de classes desconhecidas
      num_test_u = numel(obj.idx_test_u);
      obj.idx_test_u = obj.idx_test_u(randperm(num_test_u));
      
      % Complementa o conjunto de teste com 20% dos dados de classes
      % desconhecidas destinados para testes
      idx_test = [idx_test;obj.idx_test_u(1:floor(0.2*num_test_u))];
    end
        
    function [xtrain,ytrain,xval,yval] = dataTrainVal(obj,idx_train,idx_val)
      % ----------------------------------------------------------------------------------
      % This method returns training and validation samples.
      %
      % Input args
      %   idx_train: training indices.
      %   idx_val: validation indices.
      %
      % Output args
      %   xtrain: training samples.
      %   ytrain: training labels.
      %   xval: validation samples.
      %   yval: validation labels.
      % ----------------------------------------------------------------------------------
      xtrain = obj.X(idx_train,:);
      ytrain  = obj.Y(idx_train);
      xval = obj.X(idx_val,:);
      yval  = obj.Y(idx_val);
      outliers = sum(yval~=obj.trained',2) == numel(obj.trained);
      yval(outliers) = -1;
    end
        
    function [xtest,ytest] = dataTest(obj,idx_test)
      % ----------------------------------------------------------------------------------
      % This method returns test samples.
      %
      % Input args
      %   idx_test: test indices.
      %
      % Output args
      %   xtest: test samples.
      %   ytest: test labels.
      % ----------------------------------------------------------------------------------
      xtest = obj.X(idx_test,:);
      ytest  = obj.Y(idx_test);
      outliers = sum(ytest~=obj.trained',2) == numel(obj.trained);
      ytest(outliers) = -1;
    end
        
    function [xtrain,ytrain] = dataTrain(obj,idx_train)
      % ----------------------------------------------------------------------------------
      % This method returns training samples.
      %
      % Input args
      %   idx_train: training indices.
      %
      % Output args
      %   xtrain: training samples.
      %   ytrain: training labels.
      % ----------------------------------------------------------------------------------
      xtrain = obj.X(idx_train,:);
      ytrain  = obj.Y(idx_train);
    end
  end
end

