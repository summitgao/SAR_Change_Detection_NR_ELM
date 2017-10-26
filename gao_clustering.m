function CM = gao_clustering(feat_vec, ylen, xlen)
    options = [2.0; 100; 1e-5; 0];
    CMk0 = zeros(ylen, xlen);

    % feature vectors are divided into three categories by using FCM
    fprintf(' ... .. first round clustering  .... ... \n');
    [center,U,obj_fcn] = fcm(feat_vec,3, options);
    
    maxU = max(U);
    m(1) = mean(feat_vec(find(U(1,:) == maxU)));
    m(2) = mean(feat_vec(find(U(2,:) == maxU)));
    m(3) = mean(feat_vec(find(U(3,:) == maxU)));
    if m(1) == max(m)
        ttr = numel(find(U(1,:) == maxU))/(ylen*xlen)*1.15;
        ttl = numel(find(U(1,:) == maxU))/(ylen*xlen)/1.10;
    elseif m(2) == max(m)
        ttr = numel(find(U(2,:) == maxU))/(ylen*xlen)*1.15;
        ttl = numel(find(U(2,:) == maxU))/(ylen*xlen)/1.10;
    elseif m(3) == max(m)
        ttr = numel(find(U(3,:) == maxU))/(ylen*xlen)*1.15;
        ttl = numel(find(U(3,:) == maxU))/(ylen*xlen)/1.10;        
    end
    
    fprintf(' ... .. first round clustering finished!!\n');
    
    % 这里使用FCM分类五类
    c_num = 7;
    fprintf(' ... .. second round clustering  .... ...\n');
    [center,U,obj_fcn] = fcm(feat_vec,c_num, options);
    fprintf(' ... .. second round clustering finished!\n');
    maxU = max(U);
    
    for i = 1:c_num
        index{i} = find(U(i,:) == maxU);    
    end
    for i = 1:c_num
        idx_mean(i) = mean(feat_vec(index{i}));
    end    
    % 排序
    [idx_mean, idx] = sort(idx_mean);    
    % 分别计算 idx 的个数
    for i = 1:c_num
        idx_num(i) = numel(index{idx(i)});
    end    
    
    CMk0(index{idx(c_num)}) = 0.0;
    c = idx_num(c_num);
    mid_lab = 0;    

    for i = 1:c_num-1
        c = c+idx_num(c_num-i);
        if c / (ylen*xlen) < ttl
            CMk0(index{idx(c_num-i)}) = 0.0;
        elseif c / (ylen*xlen) >= ttl && c / (ylen*xlen) < ttr
            CMk0(index{idx(c_num-i)}) = 0.5;
            mid_lab = 1;
        elseif mid_lab == 0
            CMk0(index{idx(c_num-i)}) = 0.5;
            mid_lab = 1;
        else
            CMk0(index{idx(c_num-i)}) = 1;
        end
    end  
    
    CM = reshape(CMk0, ylen, xlen);
    
end











