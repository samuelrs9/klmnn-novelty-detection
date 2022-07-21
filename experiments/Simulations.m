classdef Simulations < handle
    methods(Static)        
        % Base com três linhas horizontais
        function out = horizontalLines(out_dir)
            X = []; 
            y = [];
            
            % Cria pontos de teste em um grid uniforme
            dim = 300;
            [xx,yy] = meshgrid(linspace(-1.0,1.0,dim),linspace(-1,1,dim));
            xtest = [xx(:),yy(:)];
            ytest = -ones(size(xtest,1),1);
            h = abs(yy(1)-yy(2));
            
            shift = 0.05;
            
            % Classe 1
            c1 = floor(dim/2 + 1) - floor(shift*dim);
            n_points_1 = 25;
            x1 = linspace(-1,1,n_points_1)';
            X = cat(1,X,[x1,yy(c1,1)*ones(n_points_1,1)]);
            y = cat(1,y,ones(n_points_1,1));

            % Classe 2
            c2 = floor(dim/2 + 1);
            n_points_2 = 50;
            x2 = linspace(-1,1,n_points_2)';
            X = cat(1,X,[x2,yy(c2,1)*ones(n_points_2,1)]);
            y = cat(1,y,2*ones(n_points_2,1));

            % Classe 3
            c3 = floor(dim/2 + 1) + floor(shift*dim);
            n_points_3 = 25;
            x3 = linspace(-1,1,n_points_3)';
            X = cat(1,X,[x3,yy(c3,1)*ones(n_points_3,1)]);
            y = cat(1,y,3*ones(n_points_3,1));

            % Classe de outliers aleatórios
            n_points_4 = 200;
            x4 = [2*rand(n_points_4,1)-1,0.5*rand(n_points_4,1)-0.25];
            
            outlier_1 = abs(x4(:,2)-yy(c1,1)) > 0.5*h;
            outlier_2 = abs(x4(:,2)-yy(c2,1)) > 0.5*h;
            outlier_3 = abs(x4(:,2)-yy(c3,1)) > 0.5*h;  
            
            outlier = outlier_1 & outlier_2 & outlier_3;
            
            X = cat(1,X,x4(outlier,:));
            y = cat(1,y,-1*ones(sum(outlier),1));

            % Classifica pontos do grid
            class_1 = abs(xtest(:,2)-yy(c1,1)) < 0.5*h;
            class_2 = abs(xtest(:,2)-yy(c2,1)) < 0.5*h;
            class_3 = abs(xtest(:,2)-yy(c3,1)) < 0.5*h;
            
            ytest(class_1) = 1;
            ytest(class_2) = 2;
            ytest(class_3) = 3;

            ytrain = y(y~=-1);
            xtrain = X(y~=-1,:);
            
            out.X = X; 
            out.y = y;
            out.xtrain = xtrain;
            out.ytrain = ytrain;
            out.xtest = xtest;
            out.ytest = ytest;
            
            save(strcat(out_dir,'/base.mat'),'X','y','xtrain','ytrain','xtest','ytest');
            
            %color = [0.2,0.2,0.2];
            %xlimits = [-1.0,1.0]; 
            %ylimits = [-0.5,0.5];
                        
            %figure;
            %Util.plotDecisionBoundary(xtest,ytest,color,xlimits,ylimits);
            %Util.plotClassesAux(X,y);            
        end       
        
        % Base com parábolas
        function out = parabolas(out_dir)
            X = [];
            y = [];

            % Cria pontos de teste em um grid uniforme
            dim = 200;
            [xx,yy] = meshgrid(linspace(-1,1,floor(2.5*dim)),linspace(-1,1,dim));
            grid = [xx(:),yy(:)];
            h = abs(yy(1)-yy(2));  
                        
            % Classe 1
            c1 = floor(dim/2 + 1) - 20;
            n_points_1 = 25;
            x1 = linspace(-1,1,n_points_1)';
            X = cat(1,X,[x1,x1.^2 + yy(c1,1)]);
            y = cat(1,y,ones(n_points_1,1));

            % Classe 2
            c2 = floor(dim/2 + 1);
            n_points_2 = 50;
            x2 = linspace(-1,1,n_points_2)';
            X = cat(1,X,[x2,x2.^2 + yy(c2,1)]);
            y = cat(1,y,2*ones(n_points_2,1));
            
            % Classe 3
            c3 = floor(dim/2 + 1) + 20;
            n_points_3 = 25;
            x3 = linspace(-1,1,n_points_3)';
            X = cat(1,X,[x3,x3.^2 + yy(c3,1)]);            
            y = cat(1,y,3*ones(n_points_3,1));

            % Classe de outliers aleatórios
            n_points_4 = 200;
            x4 = [2*rand(n_points_4,1)-1,0.6*rand(n_points_4,1)-0.3];
            
            outlier_1 = abs(x4(:,2) - yy(c1,1)) > 0.5*h;
            outlier_2 = abs(x4(:,2) - yy(c2,1)) > 0.5*h;
            outlier_3 = abs(x4(:,2) - yy(c3,1)) > 0.5*h;           
            
            outlier = outlier_1 & outlier_2 & outlier_3;
            
            x4 = x4(outlier,:);                        
            x4 = [x4(:,1),x4(:,1).^2 + x4(:,2)];
            
            X = cat(1,X,x4);
            y = cat(1,y,-1*ones(sum(outlier),1));
                
            ytrain = y(y~=-1);
            xtrain = X(y~=-1,:);
            
            %figure; Util.plotClasses(X,y);
            
            % Cria pontos de teste
            x = grid(:,1);
            xp = 1./(1+exp(-1.1*x))-0.5;
            xp = xp./max(xp);
            xtest = [xp,xp.^2 + grid(:,2)];            
            ytest = -ones(size(grid,1),1);
            
            % Classifica pontos de teste
            class_1 = abs(xtest(:,2) - (xtest(:,1).^2 + yy(c1,1))) < 0.5*h;
            class_2 = abs(xtest(:,2) - (xtest(:,1).^2 + yy(c2,1))) < 0.5*h;
            class_3 = abs(xtest(:,2) - (xtest(:,1).^2 + yy(c3,1))) < 0.5*h;                        
            
            ytest(class_1) = 1;
            ytest(class_2) = 2;
            ytest(class_3) = 3;
                        
            out.X = X;
            out.y = y;
            out.xtrain = xtrain;
            out.ytrain = ytrain;
            out.xtest = xtest;
            out.ytest = ytest;
            
            save(strcat(out_dir,'/base.mat'),'X','y','xtrain','ytrain','xtest','ytest');
            
            %color = [0.2,0.2,0.2];
            %xlimits = [-1.0,1.0]; 
            %ylimits = [-0.3,1.2];            
            
            %figure;
            %Util.plotDecisionBoundary(xtest,ytest,color,xlimits,ylimits);
            %Util.plotClassesAux(X,y);
        end
        
        % Base com círculos concêntricos
        function out = concentricCircles(out_dir)
            X = []; y = [];
            
            dim = 160;
            c1 = floor(dim/2 + 1) - 10;
            c2 = floor(dim/2 + 1);
            c3 = floor(dim/2 + 1) + 10;

            % Cria pontos de teste            
            r = linspace(0,0.9,dim);
            theta = cell(dim,1);
            xtest = [];
            ytest = [];
            
            rng('default');
            
            for i=1:dim
                n = 8*i;
                t0 = rand;
                theta{i} = linspace(t0,t0 + 2*pi,n+1)';
                theta{i} = theta{i}(1:end-1);
                xtest = cat(1,xtest,[r(i)*cos(theta{i}),r(i)*sin(theta{i})]);
                if i==c1
                    ytest = cat(1,ytest,1*ones(n,1));
                elseif i==c2
                    ytest = cat(1,ytest,2*ones(n,1));
                elseif i==c3
                    ytest = cat(1,ytest,3*ones(n,1));
                else
                    ytest = cat(1,ytest,-1*ones(n,1));
                end
            end
            h = abs(r(1)-r(2));
                        
            % Classe 1            
            n_points_1 = 25;
            r1 = r(c1);
            t0 = rand;
            theta1 = linspace(t0,t0 + 2*pi,n_points_1+1)';
            theta1 = theta1(1:end-1);
            X = cat(1,X,[r1*cos(theta1),r1*sin(theta1)]);
            y = cat(1,y,ones(n_points_1,1));

            % Classe 2            
            n_points_2 = 50;
            r2 = r(c2);
            t0 = rand;
            theta2 = linspace(t0,t0 + 2*pi,n_points_2+1)';
            theta2 = theta2(1:end-1);
            X = cat(1,X,[r2*cos(theta2),r2*sin(theta2)]);
            y = cat(1,y,2*ones(n_points_2,1));
            
            % Classe 3
            n_points_3 = 25;
            r3 = r(c3);
            t0 = rand;
            theta3 = linspace(t0,t0 + 2*pi,n_points_3+1)';
            theta3 = theta3(1:end-1);
            X = cat(1,X,[r3*cos(theta3),r3*sin(theta3)]);
            y = cat(1,y,3*ones(n_points_3,1));
            
            % Classe de outliers aleatórios
            n_points_4 = 200;
            r4 = 0.5*sqrt(rand(n_points_4,1))+0.2;
            theta4 = 2*pi*rand(n_points_4,1)-pi;
            
            outlier_1 = abs(r4 - r1) > 0.5*h;
            outlier_2 = abs(r4 - r2) > 0.5*h;
            outlier_3 = abs(r4 - r3) > 0.5*h;           
            
            outlier = outlier_1 & outlier_2 & outlier_3;
            
            r4 = r4(outlier);
            theta4 = theta4(outlier);
            
            X = cat(1,X,[r4.*cos(theta4),r4.*sin(theta4)]);
            y = cat(1,y,-1*ones(sum(outlier),1));
            
            ytrain = y(y~=-1);
            xtrain = X(y~=-1,:);

            out.X = X;
            out.y = y;
            out.xtrain = xtrain;
            out.ytrain = ytrain;
            out.xtest = xtest;
            out.ytest = ytest;
            
            save(strcat(out_dir,'/base.mat'),'X','y','xtrain','ytrain','xtest','ytest');
            
            %color = [0.2,0.2,0.2];
            %xlimits = [-0.8,0.8]; 
            %ylimits = [-0.8,0.8];
            
            %figure;
            %Util.plotDecisionBoundary(xtest,ytest,color,xlimits,ylimits);
            %Util.plotClassesAux(X,y);
        end        
        
        % Base com distribuições uniformes
        function out = uniformDistributions(n_points,n_dim)
            n_classes = 4;
            n_test = 10000;
            n_exp = 5;
                                    
            rng('default');
            
            % Classe 1            
            scale1 = 0.9;
            C1 = zeros(1,n_dim);
            C1(1,[1,2]) = [1.0,0.5];
            X1_train = scale1*rand(floor(n_points/n_classes),n_dim)-0.5*scale1 + C1;
            X1_test = scale1*rand(n_test,n_dim,n_exp)-0.5*scale1 + C1;
            
            % Classe 2
            scale2 = 0.9;
            C2 = zeros(1,n_dim);
            C2(1,[1,2]) = [-0.5,1.0];            
            X2_train = scale2*rand(floor(n_points/n_classes),n_dim)-0.5*scale2 + C2;
            X2_test = scale2*rand(n_test,n_dim,n_exp)-0.5*scale2 + C2;
            
            % Classe 3
            scale3 = 0.9;
            C3 = zeros(1,n_dim);
            C3(1,[1,2]) = [-1.0,-0.5];
            X3_train = scale3*rand(floor(n_points/n_classes),n_dim)-0.5*scale3 + C3;
            X3_test = scale3*rand(n_test,n_dim,n_exp)-0.5*scale3 + C3;
            
            % Classe 4
            scale4 = 0.9;
            C4 = zeros(1,n_dim);
            C4(1,[1,2]) = [0.5,-1.0];
            X4_train = scale4*rand(floor(n_points/n_classes),n_dim)-0.5*scale4 + C4;
            X4_test = scale4*rand(n_test,n_dim,n_exp)-0.5*scale4 + C4;
            
            % Classe de outliers
            scale5 = 0.9;
            C5 = zeros(1,n_dim);
            C5(1,[1,2]) = [0.0,0.0];
            X_out_val = scale5*rand(floor(n_points/n_classes),n_dim)-0.5*scale5 + C5;
            X_out_test = scale5*rand(n_test,n_dim,n_exp)-0.5*scale5 + C5;
                        
            xtrain = cat(1,X1_train,X2_train,X3_train,X4_train);
            ytrain = cat(1,ones(size(X1_train,1),1),2*ones(size(X2_train,1),1),...
                      3*ones(size(X3_train,1),1),4*ones(size(X4_train,1),1));
                  
            xtest = cat(1,X1_test,X2_test,X3_test,X4_test,X_out_test);
            ytest = cat(1,ones(size(X1_test,1),1),2*ones(size(X2_test,1),1),...
                          3*ones(size(X3_test,1),1),4*ones(size(X4_test,1),1),...
                          -1*ones(size(X_out_test,1),1));
                                                      
            X = cat(1,xtrain,X_out_val);
            y = cat(1,ytrain,-1*ones(size(X_out_val,1),1));
                              
            out.X = X; 
            out.y = y;
            out.xtrain = xtrain;
            out.ytrain = ytrain;
            out.xtest = xtest;
            out.ytest = ytest;            
            
            figure;            
            Util.plotClassesAux(X,y,[-2.3,1.6]);
            legend('location','northwest');
        end                
    end        
end

