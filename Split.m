classdef Split < handle
  methods(Static)
    
    function [idx_train,idx_test] = trainTestIdx(X,Y,training_ratio,numLabels,istrainedclass)
      numInst = size(X,1);
      n_per_class=zeros(1,numLabels);
      idx = randperm(numInst);
      
      for i=1:numInst
        n_per_class(Y(i))=n_per_class(Y(i))+1;
      end
      train_per_class=zeros(1,numLabels);
      numTrain=0;
      test_samples_untrained=zeros(1,numLabels);
      for k=1:numLabels
        if ~istrainedclass(k)
          train_per_class(k)=0;
          test_samples_untrained(k)=(n_per_class(k)-floor(n_per_class(k)*training_ratio));
          continue;
        end
        aux=floor(n_per_class(k)*training_ratio);
        train_per_class(k)=aux;
        numTrain=numTrain+aux;
      end
      
      inserted_per_class=zeros(1,numLabels);
      counter = 0;
      counter_test=0;
      i=1;
      idx_train = zeros(1,numTrain);
      idx_test=zeros(1,numInst-numTrain);
      counter_untrained=zeros(1,numLabels);
      while i<=numInst
        curr_label=Y(idx(i));
        if(inserted_per_class(curr_label)<train_per_class(curr_label))
          counter=counter+1;
          idx_train(counter)=idx(i);
          inserted_per_class(Y(idx(i)))=inserted_per_class(Y(idx(i)))+1;
        else
          if ~istrainedclass(curr_label)
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
    
    function [trainedclasses,testedclasses,istrainedclass] = selectClasses(numclasses,untrained_classes)
      idx = randperm(numclasses);
      istrainedclass = true(numclasses,1);
      for i=1:untrained_classes
        istrainedclass(idx(i))=false;
      end
      testedclasses = idx(1:untrained_classes);
      trainedclasses = idx(untrained_classes+1:numclasses);
    end
  end
end

