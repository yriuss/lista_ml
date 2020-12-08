function [d] = distance(x1,x2)
    d = sqrt(sum((x1-x2)^2))
endfunction

function [sig] = sigmoid(x)
    sig = zeros(length(x),1)
    
    for i = 1:length(x)
        sig(i) = 1/(1+exp(-x(i)))
    end
    //disp(sig)
endfunction

function [cluster,centroids] = k_means(k, max_iters, X)
    //recebendo a quantidade de dados (row) e a dimensão dos dados (column)
    [row,column] = size(X)
  
  centroids  = []
  
  //gerando os centroides
  
  centroids = [[14.75 1.73 2.39 11.4 91 3.1 3.69 .43 2.81 5.4 1.25 2.73 1150];[12.7 3.87 2.4 23 101 2.83 2.55 .43 1.95 2.57 1.19 3.13 463];...
  [13.73 4.36 2.26 22.5 88 1.28 .47 .52 1.15 6.62 .78 1.75 520]; [10 4.36 2.26 22.5 88 1.28 .47 .52 1.15 6.62 .78 1.75 520]; [10 4.36 0 22.5 88 1.28 .47 .52 1.15 6.62 .78 1.75 520]]
  
  
  /*
  //gerar randomicamente
  for i = 1:k,
      centroids = cat(1,centroids,X(grand(1,1,'uin',1,row),:))
  end*/
  
  converged = %f
  current_iter = 0
  
  //inicializando o algorítmo k-means. Ele para se tiver convergido ou o max
  //de iterações for atingido
  while ~converged | current_iter < max_iters,
      distances = zeros(1,length(centroids(:,1)))
      //inicializar clusters
      cluster = zeros(1,length(X(1,:)))
      
      for i = 1:length(X(:,1)),
          
          for c = 1:k,
              distances(c) = distance(X(i,:),centroids(c,:))
          end
          [min_value, arg_min] = min(distances(:))
          
          cluster(i) = arg_min
          
      end
      
      //inicializa a variável que irá determinar a convergência
      prev_centroids = centroids
      
      meann = zeros(1,length(k))
      
      //atualizar centroids
         
      for i = 1:k,
          
          _sum = zeros(1,length(X(1,:)))
          n = 0
          for j = 1:length(X(:,1)),
              
              if cluster(j) == i,
                  
                  _sum = _sum + X(j,:)
                  n = n + 1
              end
          end
          meann = _sum/n
          
      end
      
      
      converged = sum(prev_centroids) == sum(centroids)
      current_iter = current_iter + 1
      
  end
endfunction

function [one_hot] = to_one_hot(y, K)
    one_hot = -ones(length(y), K)
    
    for n = 1:length(y)
        for k = 1:K
            if y(n) == 1 & k == 1 then
                one_hot(n,k) = 1
            elseif y(n) == 2 & k == 2 then
                one_hot(n,k) = 1
            elseif y(n) == 3 & k == 3 then
                one_hot(n,k) = 1
            end 
        end
    end
endfunction



function [w, centroids, gama] = RBF_train(X, y)
    N = length(X(:,1))
    K = 5
    phi = zeros(N,K)
    
    [cluster, centroids] = k_means(K,100,M)
    
    maxi = 0
    for i = 1:K
        for j = 1:K
            d = distance(centroids(i,:),centroids(j,:))
            if d>maxi then
                maxi = d
            end
        end
    end
    
    
    sigma = maxi/sqrt(2*K)
    gama = 1/(sqrt(2)*sigma)^2
    for n = 1:N,
        for k = 1:K,
            phi(n,k) = exp(-(distance(X(n,:), centroids(k,:))^2)*gama)
        end
    end
    
    one_hot = to_one_hot(y, K)
    
    w = inv((phi'*phi))*phi'*one_hot
    //disp(w)
    
endfunction

function [h] = classification(sig)
    h = zeros(length(sig),1)
    //disp(sig)
    for i = 1:length(sig)
        if sig(i) >= 0.5 then
            h(i) = 1
        end
    end
endfunction

function [prediction] = RBF_pred(x, w, centroids,gama)
    K = length(centroids(:,1))
    n1 = zeros(K, 1)
    
    for k = 1:K
        n1(k) = exp(-distance(x, centroids(k,:))^2*gama)
    end
    //disp(n1'*w )
    
    [_,prediction] = max(classification(sigmoid(n1'*w )))
    //disp(prediction)
endfunction

M = csvRead("wine.csv")
y = M(2:$,1)
M = M(2:$,2:$)

new_order = grand(1, "prm", 1:length(M(:,1)))

M = M(new_order,:)
y = y(new_order)

M_1 = M(1:35,:)
M_2 = M(36:70,:)
M_3 = M(71:105,:)
M_4 = M(106:141,:)
M_5 = M(142:178,:)

y_1 = y(1:35,:)
y_2 = y(36:70,:)
y_3 = y(71:105,:)
y_4 = y(106:141,:)
y_5 = y(142:178,:)
//setosa == 1 ; versicolor == 2 ; virginica == 3


[w, centroids, gama] = RBF_train(M(36:$,:), y(36:$))

sum_pred1 = 0
sum_pred2 = 0
sum_pred3 = 0

right_pred1 = 0
right_pred2 = 0
right_pred3 = 0
right = 0

M= M_1
y= y_1
for i = 1:length(M(:,1))
    
    pred = RBF_pred(M(i,:), w, centroids,gama)
    
    if y(i) == pred then
        right = right + 1
        if pred == 1 then
            right_pred1 = right_pred1 + 1
        end
        
        if pred == 2 then
            right_pred2 = right_pred2 + 1
        end
        
        if pred == 3 then
            right_pred3 = right_pred3 + 1
        end
    end
    
    if y(i) == 1 then
        sum_pred1 = sum_pred1 + 1
    end
    if y(i) == 2 then
        sum_pred2 = sum_pred2 + 1
    end
    if y(i) == 3 then
        sum_pred3 = sum_pred3 + 1
    end
end
disp("para 1:")
disp(right_pred1/sum_pred1)
disp("para 2:")
disp(right_pred2/sum_pred2)
disp("para 3:")
disp(right_pred3/sum_pred3)
disp(right/length(y))
//disp(cluster)

