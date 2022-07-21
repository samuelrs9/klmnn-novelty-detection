classdef Plots
  
  methods(Static)
    function plotClasses(arg1,arg2)
      
      if nargin==1
        data = arg1;
      elseif nargin==2
        arg2(arg2==-1) = max(arg2)+1;
        y = unique(arg2);
        data = cell(numel(y),1);
        for i=1:numel(y)
          data{i}.X = arg1(arg2==y(i),:);
          data{i}.y = y(i);
        end
      end
      
      point_size = 50;
      hold on;
      str_legend = cell(size(data));
      h = zeros(size(data));
      marker = {'o','s','^','d','*'};
      
      n_classes = length(data);
      
      for i=1:n_classes
        if data{i}.y==n_classes % outlier
          str_legend{i} = "novelty";
          if size(data{i}.X,2)==2
            h(i) = scatter(data{i}.X(:,1),data{i}.X(:,2),0.5*point_size,marker{5});
          elseif size(data{i}.X,2)>=3
            h(i) = scatter3(data{i}.X(:,1),data{i}.X(:,2),data{i}.X(:,3),0.5*point_size,marker{5});
          end
        else % classe
          str_legend{i} = strcat("class ",int2str(data{i}.y));
          if size(data{i}.X,2)==2
            h(i) = scatter(data{i}.X(:,1),data{i}.X(:,2),point_size,'fill',marker{data{i}.y},'MarkerEdgeColor','k');
          elseif size(data{i}.X,2)>=3
            h(i) = scatter3(data{i}.X(:,1),data{i}.X(:,2),data{i}.X(:,3),point_size,'fill',marker{data{i}.y},'MarkerEdgeColor','k');
          end
        end
      end
      hold off;
      %axis equal;
      legend(h,str_legend,'fontname','times','fontsize',14);
      set(gca,'fontname','times','fontsize',14)
    end
    
    function plotClassesAux(xtrain,ytrain,xlimits,ylimits)
      if nargin==2
        xlimits = [-1.0,1.0];
        ylimits = [-1.0,1.0];
        color = [0.8,0.8,0.8];
      end
      
      hold on;
      
      set(gca,'ColorOrderIndex',1);
      
      Util.plotClasses(xtrain,ytrain);
      
      axis equal;
      xlim(xlimits);
      %ylim(ylimits);
      %set(gca,'xtick',[]);
      %set(gca,'ytick',[]);
      box on;
      
      set(gca,'fontname','times','fontsize',14);
      
      hold off;
      
      drawnow;
    end
    
    function plotDecisionBoundary(xtest,predictions,color,xlimits,ylimits)
      if nargin==2
        xlimits = [-1.0,1.0];
        ylimits = [-1.0,1.0];
        color = [0.6,0.6,0.6];
      end
      
      hold on;
      
      xtest_inlier = xtest(predictions~=-1,:);
      inlier = gscatter(xtest_inlier(:,1),xtest_inlier(:,2));
      inlier.Color = color;
      inlier.MarkerSize = 5;
      
      axis equal;
      xlim(xlimits);
      %ylim(ylimits);
      set(gca,'xtick',[]);
      set(gca,'ytick',[]);
      legend off;
      box on;
      
      set(gca,'fontname','times','fontsize',11);
      
      hold off;
      
      drawnow;
    end
    
    function viewHyperparameterCalibration(arg1,arg2,arg3,arg4)
      if nargin==3
        metric_name = arg1;
        hyperparameter = arg2;
        metric = arg3;
        figure;
        plot(hyperparameter,metric);
        xlabel('hyperparameter');
        ylabel(lower(metric_name));
        title(upper(metric_name));
      elseif nargin == 4
        metric_name = arg1;
        threshold = arg2;
        kernel = arg3;
        metric = arg4;
        figure;
        pcolor(threshold,kernel,metric);
        colorbar;
        axis([threshold(1),threshold(end),kernel(1),kernel(end)]);
        xlabel('threshold');
        ylabel('kernel');
        title(upper(metric_name));
      end
    end
    
    function metrics = compareMetricsPerTest(out_dir,method,K,kappa)
      switch method
        case {'knn','lmnn'}
          result = load(sprintf('%s/K=%d kappa=%d/%s_experiments',out_dir,K,kappa,method));
          % AFR
          afr_per_test = result.all_metrics.AFR(:,result.model.best_threshold_id);
          afr_per_test = afr_per_test(:);
          % MCC
          mcc_per_test = result.all_metrics.MCC(:,result.model.best_threshold_id);
          mcc_per_test = mcc_per_test(:);
        case {'klmnn'}
          result = load(sprintf('%s/K=%d kappa=%d/%s_experiments',out_dir,K,kappa,method));
          % AFR
          afr_per_test = result.all_metrics.AFR(result.model.best_kernel_id,result.model.best_threshold_id,:);
          afr_per_test = afr_per_test(:);
          % MCC
          mcc_per_test = result.all_metrics.MCC(result.model.best_kernel_id,result.model.best_threshold_id,:);
          mcc_per_test = mcc_per_test(:);
        case {'one_svm'}
          result = load(sprintf('%s/%s_experiments',out_dir,method));
          % AFR
          afr_per_test = result.all_metrics.AFR(:,result.model.best_kernel_id);
          afr_per_test = afr_per_test(:);
          % MCC
          mcc_per_test = result.all_metrics.MCC(:,result.model.best_kernel_id);
          mcc_per_test = mcc_per_test(:);
        case {'knfst','multi_svm','kpca'}
          result = load(sprintf('%s/%s_experiments',out_dir,method));
          % AFR
          afr_per_test = result.all_metrics.AFR(result.model.best_kernel_id,result.model.best_threshold_id,:);
          afr_per_test = afr_per_test(:);
          % MCC
          mcc_per_test = result.all_metrics.MCC(result.model.best_kernel_id,result.model.best_threshold_id,:);
          mcc_per_test = mcc_per_test(:);
      end
      metrics.mcc_per_test = mcc_per_test;
      metrics.afr_per_test = afr_per_test;
    end
    
    function boxPlotMetricsPerMethod(out_dir,methods,K,kappa)
      mcc = [];
      afr = [];
      for i=1:numel(methods)
        metrics = Util.compareMetricsPerTest(out_dir,methods{i},K,kappa);
        mcc = cat(2,mcc,metrics.mcc_per_test);
        afr = cat(2,afr,metrics.afr_per_test);
      end
      figure;
      boxplot(mcc,'Labels',methods);
      xlabel('Methods');
      ylabel('Matthews correlation coefficient (MCC)');
      
      figure;
      boxplot(afr,'Labels',methods);
      xlabel('Methods');
      ylabel('Average false rate (AFR)');
    end
    
    function plotDecisionBoundaryMethods()
      close all;
      
      %out_dir = {'simulation_3'};
      out_dir = {'simulation_1','simulation_2','simulation_3'};
      
      methods = {'one_svm'};
      %methods = {'training','knn','lmnn','klmnn','knfst','one_svm','multi_svm','kpca'};
      
      n_methods = numel(methods);
      n_sim = numel(out_dir);
      
      for i=1:n_sim
        switch out_dir{i}
          % Carrega a base da simulação
          case 'simulation_1'
            load('simulation_1/base.mat');
            K = 2;
            kappa = 1;
            xlimits = [-1.1,1.1];
            ylimits = [-0.5,0.5];
            color = [0.6,0.6,0.6];
          case 'simulation_2'
            load('simulation_2/base.mat');
            K = 1;
            kappa = 1;
            xlimits = [-1.1,1.1];
            ylimits = [-0.3,1.2];
            color = [0.6,0.6,0.6];
          case 'simulation_3'
            load('simulation_3/base.mat');
            K = 2;
            kappa = 1;
            xlimits = [-0.8,0.8];
            ylimits = [-0.8,0.8];
            color = [0.6,0.6,0.6];
        end
        
        for j=1:n_methods
          id = sub2ind([n_sim,n_methods],i,j);
          figure(id);
          set(gcf,'Position',[200 200 600 400]);
          
          %subplot(n_methods,n_sim,id);
          
          switch methods{j}
            case 'training'
              try
                Util.plotClassesAux(X,y,xlimits,ylimits);
                if strcmp(out_dir{i},'simulation_1')
                  str_sim = 'horizontal lines';
                elseif strcmp(out_dir{i},'simulation_2')
                  str_sim = 'parabolas';
                elseif strcmp(out_dir{i},'simulation_3')
                  str_sim = 'concentric circles';
                end
                %title(str_sim,'fontname','times','fontsize',11);
                if strcmp(out_dir{i},'simulation_1')
                  %ylabel('training');
                end
                
              catch
                scatter(rand(id,1),rand(id,1))
                fprintf('\ntraining error\n');
              end
            case 'knn'
              try
                knn_dir = strcat(out_dir{i},'/K=',int2str(K),' kappa=',int2str(kappa));
                load(strcat(knn_dir,'/knn_predictions.mat'));
                
                Util.plotDecisionBoundary(xtest,predictions,color,xlimits,ylimits);
                %title(methods{j});
                if strcmp(out_dir{i},'simulation_1')
                  %ylabel('knn');
                end
              catch
                scatter(rand(id,1),rand(id,1))
                fprintf('\nknn error\n');
              end
            case 'lmnn'
              try
                knn_dir = strcat(out_dir{i},'/K=',int2str(K),' kappa=',int2str(kappa));
                load(strcat(knn_dir,'/lmnn_predictions.mat'));
                
                Util.plotDecisionBoundary(xtest,predictions,color,xlimits,ylimits);
                %title(methods{j});
                if strcmp(out_dir{i},'simulation_1')
                  %ylabel('lmnn');
                end
              catch
                scatter(rand(id,1),rand(id,1));
                fprintf('\nlmnn error\n');
              end
            case 'klmnn'
              try
                knn_dir = strcat(out_dir{i},'/K=',int2str(K),' kappa=',int2str(kappa));
                load(strcat(knn_dir,'/klmnn_predictions.mat'));
                
                %color = [0.2,0.2,0.2];
                Util.plotDecisionBoundary(xtest,predictions,color,xlimits,ylimits);
                %title(methods{j});
                if strcmp(out_dir{i},'simulation_1')
                  %ylabel('klmnn');
                end
              catch
                scatter(rand(id,1),rand(id,1));
                fprintf('\nklmnn error\n');
              end
            case 'knfst'
              try
                load(strcat(out_dir{i},'/knfst_predictions.mat'));
                
                %color = [0.2,0.2,0.2];
                Util.plotDecisionBoundary(xtest,predictions,color,xlimits,ylimits);
                %title(methods{j});
                if strcmp(out_dir{i},'simulation_1')
                  %ylabel('knfst');
                end
              catch
                scatter(rand(id,1),rand(id,1));
                fprintf('\nknfst error\n');
              end
            case 'one_svm'
              try
                load(strcat(out_dir{i},'/one_svm_predictions.mat'));
                
                %color = [0.8,0.8,0.8];
                Util.plotDecisionBoundary(xtest,predictions,color,xlimits,ylimits);
                %title(methods{j});
                if strcmp(out_dir{i},'simulation_1')
                  %ylabel('osvm');
                end
              catch
                scatter(rand(id,1),rand(id,1));
                fprintf('\none svm error\n');
              end
            case 'multi_svm'
              try
                load(strcat(out_dir{i},'/multi_svm_predictions.mat'));
                
                %color = [0.8,0.8,0.8];
                Util.plotDecisionBoundary(xtest,predictions,color,xlimits,ylimits);
                %title(methods{j});
                if strcmp(out_dir{i},'simulation_1')
                  %ylabel('mcsvm');
                end
              catch
                scatter(rand(id,1),rand(id,1));
                fprintf('\nmulti svm error\n');
              end
            case 'kpca'
              try
                load(strcat(out_dir{i},'/kpca_predictions.mat'));
                
                %color = [0.2,0.2,0.2];
                Util.plotDecisionBoundary(xtest,predictions,color,xlimits,ylimits);
                %title(methods{j});
                if strcmp(out_dir{i},'simulation_1')
                  %ylabel('kpcanov');
                end
              catch
                scatter(rand(id,1),rand(id,1));
                fprintf('\nkpca error\n');
              end
          end
        end
      end
    end
  end
end

