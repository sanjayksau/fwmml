function model = FWMML( X, Y, optmParameter)
   %% optimization parameters
    lambda1          = optmParameter.lambda1; % YR
    lambda2          = optmParameter.lambda2; % W'W
    lambda3          = optmParameter.lambda3; % W regularization
    lambda4          = optmParameter.lambda4; % R regularization
    lambda5          = optmParameter.lambda5; % Projected Input Space-Instance Similarity laplacian
    lambda6          = optmParameter.lambda6; % Predictions-Correlation Laplacian
    
    etaW = optmParameter.etaW; 
    etaR = optmParameter.etaR;
    J = optmParameter.J;
    
    maxIter          = optmParameter.maxIter;
    rho              = optmParameter.rho;

    num_dim   = size(X,2);  % d
    num_class = size(Y,2);  % q
    num_inst =  size(X,1);  % n
    
    
    %C = pdist2( Y'+eps, Y'+eps, 'cosine' );
    %L = diag(sum(C,2)) - C;
    %% initialization
    W_k   = (X'*X + rho*eye(num_dim)) \ (X'*Y);%zeros(num_dim,num_label)
    
    R_k = zeros(num_class,num_class); %eye(num_class,num_class);
    %R_k = 1 - squareform(pdist(Y', 'cosine'));
       
    %Feature Similarity
    %S = exp(-squareform(pdist(X')));
    iter = 1; oldloss = 9999999;
    
    %Instance similarity and Laplacian
    %Try different similarity measures 
    SI = exp(-squareform(pdist(X))); %Try GAUSSIAN
    Linst = diag(sum(SI, 2)) - SI;
    
    epsilon=0.000001;
    E = ones(size(Y));
    while iter <= maxIter
       %Update W
       HingeComp = max(E - (Y*R_k) .* (X*W_k), 0) .* (-Y*R_k);
       L = diag(sum(R_k)) - R_k;
       Q_k_num = sum(vecnorm(W_k, 2, 2)+epsilon);
       Q_k = Q_k_num / (vecnorm(W_k, 2, 2)+epsilon);
       delW = X'*(HingeComp) + 2*lambda2 * (-W_k)*(R_k - W_k'*W_k) + lambda3 * Q_k * W_k ...
           + lambda5*X'*Linst*X*W_k + lambda6*W_k*(L+L');
       
       grad = delW / norm(delW, 'fro');
       alpha = computeStepSize('W', W_k, grad, etaW, J, X, Y, W_k, R_k, Linst, L, E, lambda1, lambda2, lambda3, lambda4, lambda5, lambda6); 
       W_k = W_k - alpha * grad;
       
       HingeComp = max(E - (Y*R_k) .* (X*W_k), 0) .* (-X*W_k);
       delR = Y' * (HingeComp) + lambda1 * Y' * (Y*R_k - Y) ...
           + lambda2 * (R_k - W_k' * W_k) + lambda4 * R_k;
       
       %Update R
       L = diag(sum(R_k)) - R_k;

       grad = delR / norm(delR, 'fro');
       alpha = computeStepSize('R', R_k, grad, etaR, J, X, Y, W_k, R_k, Linst, L, E, lambda1, lambda2, lambda3, lambda4, lambda5, lambda6); 
       R_k = R_k - alpha * grad;

            
      %% Loss
       
       totalloss = ObjectiveValue( J, X, Y, W_k, R_k, Linst, L, E, ...
           lambda1, lambda2, lambda3, lambda4, lambda5, lambda6);
       loss(iter,1) = totalloss;
       iter=iter+1;
       if (iter > 5)
           if (abs(oldloss - totalloss)/totalloss < 0.0001)
                break
           else
                oldloss = totalloss;
           end
       end
    end
    fprintf('- Last Iteration: %d/%d', iter, maxIter);
    model.W = W_k;
    model.R = R_k;
    %model.loss = loss;
    plot(loss)
    model.optmParameter = optmParameter;
    model.loss = loss;
end

function loss = ObjectiveValue( J, X, Y, W_k, R_k, Linst, L, E, lambda1, lambda2, lambda3, lambda4, lambda5, lambda6)
      %% Loss
      T1 = 0.5 * norm((max(E - (Y*R_k).* (X*W_k), 0)).^2, 1);
      %T2 = 0.5 * lambda1 * trace((Y*R_k - Y)'*(Y*R_k - Y));
      T2 = 0.5 * lambda1 * norm(Y*R_k-Y, 'fro')^2;
      T3 = 0.5 * lambda2 * trace((R_k - W_k'*W_k)'*(R_k - W_k'*W_k));
      T4 = 0.5 * lambda3 * trace(W_k' * W_k);               %norm(W_k, 'fro')^2
      T5 = 0.5 * lambda4 * trace(R_k' * R_k);               %
      T6 = 0.5 * lambda5 * trace((X*W_k)'*Linst*(X*W_k));
      T7 = lambda6 * trace(W_k * L * W_k');
      loss = T1 + T2 + T3 + T4 + T5 + T6 + T7;
end

function [alpha] = computeStepSize(V, M, grad, alpha, J, X, Y, W_k, R_k, Linst, L, E, lambda1, lambda2, lambda3, lambda4, lambda5, lambda6) 
       obj1 = ObjectiveValue(J, X, Y, W_k, R_k, Linst, L, E, lambda1, lambda2, lambda3, lambda4, lambda5, lambda6);
       flag = 0; j = 1;
       while ( j > 0 )
           Mnew = M - alpha * grad;
           if V == 'W'
                obj2 = ObjectiveValue(J, X, Y, Mnew, R_k, Linst, L, E, lambda1, lambda2, lambda3, lambda4, lambda5, lambda6);
           elseif V == 'R'
               obj2 = ObjectiveValue(J, X, Y, W_k, Mnew, Linst, L, E, lambda1, lambda2, lambda3, lambda4, lambda5, lambda6);
           end
           if obj2 > obj1
               flag = 1;
               alpha = alpha * 0.5;          
           else
               break;
           end
       end 
       %if flag
       %    alpha = alpha * 2;
       %end
end

