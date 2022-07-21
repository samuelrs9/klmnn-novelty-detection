classdef SplitData < handle
  properties
    X = []
    Y = [];
    
    training_ratio = 0;
    test_ratio = 0;
    
    num_classes = 0;
    num_inst = 0;
    istrainedclass = [];
    
    classes = [];   % Classses
    trained = [];   % Classes de treinamento
    untrained = []; % Classes usadas como outliers
    
    id_trained = [];   % Índices de dados de classes "conhecidas"
    id_untrained = []; % Índices de dados "desconhecidos (outliers)
    
    id_train_val_t = []; % Corresponde a 70% dos dados de classes conhecidas
    id_test_t = [];      % Corresponde a 30% dos dados de classes conhecidas
    
    id_val_u = []; % corresponde a 50% dos dados de classes desconhecidas
    id_test_u = []; % corresponde a 50% dos dados de classes desconhecidas
  end
  methods
    % Construtor
    function obj = SplitData(X,Y,training_ratio,num_untrained_classes)
      obj.Y = Y;
      obj.X = X;
      obj.training_ratio = training_ratio;
      obj.test_ratio = 0;
      obj.classes = unique(Y);
      obj.num_classes = numel(obj.classes);
      obj.num_inst = numel(Y);
      [obj.trained,obj.untrained,obj.istrainedclass] = obj.selectClasses(num_untrained_classes);
      [obj.id_trained,obj.id_untrained] = obj.splitTrainedUntrained(obj.istrainedclass);
      [obj.id_train_val_t,obj.id_test_t,obj.id_val_u,obj.id_test_u] = obj.idTrainValTest();
    end
    
    % Seleciona as classes trained e untrained
    function [trained,untrained,istrainedclass] = selectClasses(obj,num_untrained_classes)
      if sum(obj.classes == -1) > 0
        untrained = obj.classes(obj.classes==-1);
        trained = obj.classes(obj.classes~=-1);
        istrainedclass = true(obj.num_classes,1);
        istrainedclass(obj.classes==-1) = false;
      else
        idx = randperm(obj.num_classes)';
        istrainedclass = true(obj.num_classes,1);
        istrainedclass(idx(1:num_untrained_classes)) = false;
        untrained = idx(1:num_untrained_classes);
        trained = idx(num_untrained_classes+1:obj.num_classes);
      end
    end
    
    % Divide os indices dos dados nos conjuntos trained (C_t) e untrained (C_u)
    function [id_trained,id_untrained] = splitTrainedUntrained(obj,istrainedclass)
      idx = (1:obj.num_inst)';
      id_trained = [];
      id_untrained = [];
      for i=1:obj.num_classes
        if istrainedclass(i)
          id_trained = [id_trained;idx(obj.Y==obj.classes(i))];
        else
          id_untrained = [id_untrained;idx(obj.Y==obj.classes(i))];
        end
      end
    end
    
    % Divide os indices dados em treinamento e teste
    function [id_train_val_t,id_test_t,id_val_u,id_test_u] = idTrainValTest(obj)
      % Permuta os dados de classes conhecidas
      num_trained = numel(obj.id_trained);
      obj.id_trained = obj.id_trained(randperm(num_trained));
      
      % Divide os dados de classes conhecidas em treino/validação e teste.
      num_train_val_t = floor((1-obj.test_ratio)*num_trained);
      id_train_val_t = obj.id_trained(1:num_train_val_t);
      id_test_t = obj.id_trained(num_train_val_t + 1:num_trained);
      
      % Permuta os dados de classes desconhecidas
      num_untrained = numel(obj.id_untrained);
      obj.id_untrained = obj.id_untrained(randperm(num_untrained));
      
      % Divide os dados de classes desconhecidas em validação e teste.
      num_val_u = floor((1-obj.test_ratio)*num_untrained);
      id_val_u = obj.id_untrained(1:num_val_u);
      id_test_u = obj.id_untrained(num_val_u+1:num_untrained);
    end
    
    % Separa uma parte do conjunto de treinamento para a validação.
    function [id_train,id_val] = idTrainVal(obj)
      % Permuta as amostras de treinamento/validação
      num_train_val = numel(obj.id_train_val_t);
      obj.id_train_val_t = obj.id_train_val_t(randperm(num_train_val));
      
      % Divide os dados de treinamento/validação
      num_train = floor(obj.training_ratio*num_train_val);
      num_val_t = num_train_val - num_train;
      
      id_train = obj.id_train_val_t(1:num_train);
      id_val = obj.id_train_val_t(num_train+1:num_train_val);
      
      % Permuta as amostras de validação desconhecidas
      num_val_u = numel(obj.id_val_u);
      obj.id_val_u = obj.id_val_u(randperm(num_val_u));
      
      % Complementa o conjunto de validação com uma parte dos dados de classes
      % não treinadas destinados para validação
      id_val = [id_val;obj.id_val_u(1:floor(obj.training_ratio*num_val_u))];
    end
    
    % Separa uma parte do conjunto para teste.
    function id_test = idTest(obj)
      % Permuta os dados de teste de classes conhecidas
      num_test_t = numel(obj.id_test_t);
      obj.id_test_t = obj.id_test_t(randperm(num_test_t));
      
      % Pega 50% dos dados de teste de classes conhecidas
      id_test = obj.id_test_t(1:floor(0.5*num_test_t));
      
      % Permuta os dados de teste de classes desconhecidas
      num_test_u = numel(obj.id_test_u);
      obj.id_test_u = obj.id_test_u(randperm(num_test_u));
      
      % Complementa o conjunto de teste com 20% dos dados de classes
      % desconhecidas destinados para testes
      id_test = [id_test;obj.id_test_u(1:floor(0.2*num_test_u))];
    end
    
    % Retorna os dados de treinamento e validação passados os índices
    function [xtrain,ytrain,xval,yval] = dataTrainVal(obj,id_train,id_val)
      xtrain = obj.X(id_train,:);
      ytrain  = obj.Y(id_train);
      xval = obj.X(id_val,:);
      yval  = obj.Y(id_val);
      outliers = sum(yval~=obj.trained',2) == numel(obj.trained);
      yval(outliers) = -1;
    end
    
    % Retorna os dados de teste passados os índices
    function [xtest,ytest] = dataTest(obj,id_test)
      xtest = obj.X(id_test,:);
      ytest  = obj.Y(id_test);
      outliers = sum(ytest~=obj.trained',2) == numel(obj.trained);
      ytest(outliers) = -1;
    end
    
    % Retorna os dados de treinamento passados os índices
    function [xtrain,ytrain] = dataTrain(obj,id_train)
      xtrain = obj.X(id_train,:);
      ytrain  = obj.Y(id_train);
    end
  end
end

