classdef MetricsReport < handle
  
  properties
    CM = []; % Confusion matrix
    
    FP = []; % False positive
    FN = []; % False negative
    TP = []; % True positive
    TN = []; % True negative
    
    FPR = []; % Fall out or false positive rate
    FNR = []; % False negative rate
    TPR = []; % Sensitivity, hit rate, recall, or true positive rate
    TNR = []; % Specificity or true negative rate
    
    NPV = []; % Negative predictive value
    PPV = []; % Positive predictive value
    FDR = []; % False discovery rate
    
    AFR = []; % Average false rate:
    
    ACC = []; % Overall accuracy
    MC = [];  % Combined metric
    F1 = [];  % F1 Score
    MCC = []; % Matthews correlation coefficient
  end
  
  methods
    % Construtor
    function obj = MetricsReport(arg1,arg2)
      if nargin==2
        ground_truth = arg1;
        prediction = arg2;
        % Confusion matrix
        obj.CM = confusionmat(ground_truth,prediction);
      elseif nargin==1
        obj.CM = arg1;
      end
      
      obj.FP = sum(obj.CM,1)' - diag(obj.CM);
      obj.FN = sum(obj.CM,2) - diag(obj.CM);
      obj.TP = diag(obj.CM);
      obj.TN = sum(obj.CM(:)) - (obj.FP + obj.FN + obj.TP);
      
      % Sensitivity, hit rate, recall, or true positive rate
      obj.TPR = obj.TP./(obj.TP + obj.FN);
      
      % Specificity or true negative rate
      obj.TNR = obj.TN./(obj.TN + obj.FP);
      
      % Precision or positive predictive value
      obj.PPV = obj.TP./(obj.TP + obj.FP);
      
      % Negative predictive value
      obj.NPV = obj.TN./(obj.TN + obj.FN);
      
      % Fall out or false positive rate
      obj.FPR = obj.FP./(obj.FP + obj.TN);
      
      % False negative rate
      obj.FNR = obj.FN./(obj.TP + obj.FN);
      
      % False discovery rate
      obj.FDR = obj.FP./(obj.TP + obj.FP);
      
      % Average false rate:
      obj.AFR = 0.5*(obj.FPR + obj.FNR);
      
      % Overall accuracy
      obj.ACC = (obj.TP + obj.TN)./(obj.TP + obj.FP + obj.FN + obj.TN);
      
      % Combined metric
      obj.MC = obj.TPR.*(1 - obj.FPR);
      
      % F1 Score
      obj.F1 = 2*(obj.PPV.*obj.TPR)./(obj.PPV + obj.TPR);
      
      % Matthews correlation coefficient
      obj.MCC = (obj.TP.*obj.TN - obj.FP.*obj.FN)./...
        sqrt((obj.TP + obj.FP).*(obj.TP + obj.FN).*(obj.TN + obj.FP).*(obj.TN + obj.FN));
    end
  end
end

