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
  new_order = grand(1, "prm", 1:length(M(:,1)))
  M = M(new_order, :)
  centroids = M(1:k,:)
  //disp(size(centroids))
  
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



function [w, gama] = RBF_train(X, y, centroids)
    N = length(X(:,1))
    phi = zeros(N,K + 1)
    //inserindo viés
    phi(:,K+1) = -0*ones(N,1)
    
    maxi = 0
    //calcula distancia máxima entre centroides
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
            phi(n,k) = exp(-(distance(X(n,:), centroids(k,:))^2)*gama/100)
        end
    end
    
    one_hot = to_one_hot(y, 1)
    //disp(phi)
    w = pinv((phi'*phi))*phi'*one_hot
    //disp(one_hot)
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
    n1 = zeros(K + 1, 1)
    //viés
    n1(K+1) = -0
    for k = 1:K
        n1(k) = exp(-distance(x, centroids(k,:))^2*gama/100)
    end
    //disp(n1'*w )
    
    prediction = classification(sigmoid(n1'*w ))
endfunction

function [acc] = kfolds(M, y, centroids, k)
    
    N = floor(length(M(:,1))/k)
    soma = 0
    for c = 1:k-1
        [w, gama] = RBF_train(M(c*N + 1:$,:), y(c*N + 1:$), centroids)
        
        sum_pred1 = 0
        sum_pred2 = 0
        sum_pred3 = 0
        
        right_pred1 = 0
        right_pred2 = 0
        right_pred3 = 0
        right = 0
        
        
        for i = (c-1)*N + 1:c*N
            pred = RBF_pred(M(i,:), w, centroids,gama)
            
			if y(i) == pred then
				right = right + 1
				if pred == 0 then
					right_pred1 = right_pred1 + 1
				end
				
				if pred == 1 then
					right_pred2 = right_pred2 + 1
				end
			end
			
			if y(i) == 0 then
				sum_pred1 = sum_pred1 + 1
			end
			if y(i) == 1 then
				sum_pred2 = sum_pred2 + 1
			end
		end
        //Pode descomentar abaixo para ver os acertos de cada parte
		disp("para 1:")
		disp(right_pred1/sum_pred1)
		disp("para 2:")
		disp(right_pred2/sum_pred2)
		//disp(cluster)
        soma = soma + right
    end
    
    [w, gama] = RBF_train(M((k-1)*N + 1:$,:), y(c*N + 1:$), centroids)
    
    sum_pred1 = 0
    sum_pred2 = 0
    sum_pred3 = 0
    
    right_pred1 = 0
    right_pred2 = 0
    right_pred3 = 0
    right = 0
    
    for i = (k-1)*N + 1:length(M(:,1))
            pred = RBF_pred(M(i,:), w, centroids,gama)
            
			if y(i) == pred then
				right = right + 1
				if pred == 0 then
					right_pred1 = right_pred1 + 1
				end
				
				if pred == 1 then
					right_pred2 = right_pred2 + 1
				end
			end
			
			if y(i) == 0 then
				sum_pred1 = sum_pred1 + 1
			end
			if y(i) == 1 then
				sum_pred2 = sum_pred2 + 1
			end
	end
    
    soma = soma + right
    acc = soma/length(M(:,1))
endfunction

M = csvRead("pulsos.csv")
y = M(:,length(M(1,:)))
M = M(:,1:length(M(1,:))-  1)
//disp(y)

//A ordem da matriz de dados é sempre atualizada. Eu podia ter guardado os que eu consegui melhor resultado, mas como n posso mandar arquivo csv e se eu colocasse para receber um valor aqui ficaria muito poluído, optei por deixar aleatório mesmo.Se rodar múltiplas vezes, seram obtidos resultados diferentes. O máximo que eu vi foi 93% e o mínimo, 83%
new_order = grand(1, "prm", 1:length(M(:,1)))

M = M(new_order,:)
y = y(new_order)
K = 150

//Como a matriz é muito grande, é interessante rodar o script com o k-means UMA única vez. Depois disso comentar essa função abaixo e somente rodar o resto. Na primeira vez que roda, demora um pouco mesmo...
[cluster, centroids] = k_means(K,50,M)

acc = kfolds(M, y, centroids, 5)
str_acc = "a porcentagem de acerto é: "+string(acc*100)+"%"
disp(str_acc)
