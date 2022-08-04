classdef SimpleSplit < handle
  % --------------------------------------------------------------------------------------
  % This class is used to manage a simple random splitting of dataset into
  % training andtest sets. 
  %
  % Version 2.0, July 2022.
  % By Samuel Silva (samuelrs@usp.br).
  % --------------------------------------------------------------------------------------      
  methods(Static)    
    function [idx_train,idx_test] = trainTestIdx(X,Y,training_ratio,istrained)
      % ----------------------------------------------------------------------------------
      % This method splits a part of the dataset for training and test.
      %
      % Input args
      %   X: samples [num_samples x dimension].
      %   Y: sample labels [num_samples x 1].
      %   training_ratio: training sample rate.
      %   istrained: % boolean list, true for trained classes and false for untrained
      %
      % Output args
      %   idx_train: train indices.      
      %   idx_test: test indices.      
      % ----------------------------------------------------------------------------------      
      num_classes = numel(unique(Y));
      numInst = size(X,1);
      n_per_class=zeros(1,num_classes);
      idx = randperm(numInst);
      
      for i=1:numInst
        n_per_class(Y(i))=n_per_class(Y(i))+1;
      end
      train_per_class=zeros(1,num_classes);
      numTrain=0;
      test_samples_untrained=zeros(1,num_classes);
      for k=1:num_classes
        if ~istrained(k)
          train_per_class(k)=0;
          test_samples_untrained(k)=(n_per_class(k)-floor(n_per_class(k)*training_ratio));
          continue;
        end
        aux=floor(n_per_class(k)*training_ratio);
        train_per_class(k)=aux;
        numTrain=numTrain+aux;
      end
      
      inserted_per_class=zeros(1,num_classes);
      counter = 0;
      counter_test=0;
      i=1;
      idx_train = zeros(1,numTrain);
      idx_test=zeros(1,numInst-numTrain);
      counter_untrained=zeros(1,num_classes);
      while i<=numInst
        curr_label=Y(idx(i));
        if(inserted_per_class(curr_label)<train_per_class(curr_label))
          counter=counter+1;
          idx_train(counter)=idx(i);
          inserted_per_class(Y(idx(i)))=inserted_per_class(Y(idx(i)))+1;
        else
          if ~istrained(curr_label)
            if counter_untrained(curr_label)<test_samples_untrained(curr_label)
              counter_untrained(curr_label)=counter_untrained(curr_label)+1;
              counter_test=counter_test+1;
              idx_test(counter_test) = idx(i);
            end
          else
            counter_test=counter_test+1;
            idx_test(counter_test) = idx(i);
          end
        end
        i=i+1;
      end
      idx_test = idx_test(1:counter_test);
      
      idx_train = idx_train';
      idx_test = idx_test';
    end
    
    function [xtrain,xtest,ytrain,ytest] = dataTrainTest(idx_train,idx_test,X,Y)
      % ----------------------------------------------------------------------------------
      % This method returns training and test samples.
      %
      % Input args
      %   idx_train: training indices.
      %   idx_test: validation indices.
      %   X: all samples.
      %   Y: all samples labels.
      %
      % Output args
      %   xtrain: training samples.
      %   ytrain: training labels.
      %   xtest: test samples.
      %   ytest: test labels.
      % ----------------------------------------------------------------------------------      
      numTrain=size(idx_train,1);
      numTest=size(idx_test,1);
      xtrain = X(idx_train(1:numTrain),:);
      xtest = X(idx_test(1:numTest),:);
      if ~isempty(Y)
        ytrain  = Y(idx_train(1:numTrain));
        ytest  = Y(idx_test(1:numTest));
      else
        ytrain = [];
        ytest = [];
      end
    end
    
    function [trained,untrained,istrained] = selectClasses(num_classes,untrained_classes)
      % ----------------------------------------------------------------------------------    
      % This method selects trained and untrained classes.
      %
      % Input args
      %   num_classes: number of classes in the dataset.
      %   num_untrained: if it is an integer, then num_untrained classes will be selected 
      %     randomly and all samples of those classes will be set as novelty. If it is the 
      %     string "auto", the selection will be made automatically, in this case, all 
      %     samples that have the label equal to -1 will be used as novelty.
      %
      % Output args
      %   trained: list of trained classes.
      %   untrained: list of untrained classes.
      %   istrained: list of boolean values indicating whether a class is
      %     trained or untrained.
      % ----------------------------------------------------------------------------------
      
      idx = randperm(num_classes);
      istrained = true(num_classes,1);
      for i=1:untrained_classes
        istrained(idx(i))=false;
      end
      untrained = idx(1:untrained_classes);
      trained = idx(untrained_classes+1:num_classes);
    end
  end
end

