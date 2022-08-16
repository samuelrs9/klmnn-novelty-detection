classdef Datasets < handle
  % ---------------------------------------------------------
  % Useful class for loading test datasets.
  %
  % Version 2.0, July 2022.
  % By Samuel Silva (samuelrs@usp.br).
  % ---------------------------------------------------------
  properties
    root_dir = "datasets";
  end
  
  methods
    function obj = Datasets(root_dir)
      % ---------------------------------------------------------
      % Constructor
      %
      % Input args:
      %   root_dir: root directory where the datasets are located
      % ---------------------------------------------------------
      if nargin>0
        obj.root_dir = root_dir;
      end
    end
    
    function iris = loadIris(obj)
      % ---------------------------------------------------------
      % Loads the Iris dataset
      % ---------------------------------------------------------
      fprintf('\nLoading iris dataset... ');
      data = load(obj.root_dir+"/iris/iris.mat");
      X = data.X';
      Y = data.y';
      if min(Y) == 0
        Y = Y+1;
      end
      X = X-mean(X);
      X = X/max(X(:));
      
      iris.X = X;
      iris.Y = Y;
      iris.samples_per_classe = sum(Y==unique(Y)',1);
      
      fprintf('done!\n');
    end
    
    function glass = loadGlass(obj)
      % ---------------------------------------------------------
      % Loads the Glass dataset
      % https://www.kaggle.com/datasets/uciml/glass
      % ---------------------------------------------------------
      fprintf('\nLoading glass dataset... ');
      data = load(obj.root_dir+"/glass/glass.dat");
      X = data(:,2:10);
      Y = data(:,11);
      for i=1:numel(Y)
        if(Y(i)>4)
          Y(i)=Y(i)-1;
        end
      end
      X = X-mean(X);
      X = X/max(X(:));
      
      glass.X = X;
      glass.Y = Y;
      glass.samples_per_classe = sum(Y==unique(Y)',1);
      
      fprintf('done!\n');
    end
    
    function libras = loadLibras(obj)
      % ---------------------------------------------------------
      % Loads the Brazilian Sign Language dataset created
      % using Leapmotion sensor
      % ---------------------------------------------------------
      fprintf('\nLoading libras dataset... ');
      data = load(obj.root_dir+"/libras/libras.mat");
      X = data.DATA';
      Y = data.labels';
      if min(Y) == 0
        Y = Y+1;
      end
      X = X-mean(X);
      X = X/max(X(:));
      
      libras.X = X;
      libras.Y = Y;
      libras.samples_per_classe = sum(Y==unique(Y)',1);
      
      fprintf('done!\n');
    end
    
    function body = loadBodyPoses(obj)
      % ---------------------------------------------------------
      % Loads the body pose dataset created using the
      % kinect sensor
      % ---------------------------------------------------------
      fprintf('\nLoading body poses dataset... ');
      data = load(obj.root_dir+"/body/posedb.txt");
      X = data(:,[1:4 7:10 13]);
      Y = data(:,14);
      
      X = X-mean(X);
      X = X/max(X(:));
      
      body.X = X;
      body.Y = Y;
      body.samples_per_classe = sum(Y==unique(Y)',1);
      
      fprintf('done!\n');
    end
    
    function mnist = loadMNIST(obj,num_samples)
      % ---------------------------------------------------------
      % Loads the MNIST dataset.
      %
      % Input args
      %   num_samples: number of samples to be loaded.
      % ---------------------------------------------------------
      fprintf('\nLoading mnist dataset... ');
      currfolder = pwd;
      cd(obj.root_dir+"/mnist/");
      [imgs,labels] = readMNIST('train-images.idx3-ubyte','train-labels.idx1-ubyte',num_samples,0);
      X = reshape(imgs,[],num_samples)';
      Y = labels+1;
      X = X-mean(X);
      X = X/max(X(:));
      mnist.X = X; mnist.Y = Y;
      cd(currfolder);
      fprintf('done!\n');
    end
  end
end

