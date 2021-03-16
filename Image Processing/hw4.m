close all; clear; clc

[train_images, train_labels] = mnist_parse('train-images.idx3-ubyte', 'train-labels.idx1-ubyte');
[test_images, test_labels] = mnist_parse('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte');


[a,b,c] = size(train_images);
[a2,b2,c2] = size(test_images);
train_data = zeros(a*b, c);
test_data = zeros(a2*b2, c2);
for i=1:1:c
    train_data(:,i)=reshape(train_images(:,:,i),a*b, 1);
end
for i=1:1:c2
    test_data(:,i)=reshape(test_images(:,:,i),a2*b2, 1);
end

mu = mean(train_data,2); 
train_data = train_data - repmat(mu,1,length(train_data));

[U,S,V]=svd(train_data,'econ');
proj=(S*V')';

plot(diag(S)/sum(diag(S)), '-o')
xlabel('Rank','Fontsize',12)
ylabel('Proportion Captured','Fontsize',12)

figure(3);
colormap jet
for i=0:1:9
    index = find(train_labels == i);
    scatter3(V(index,2),V(index,3),V(index,5),20,train_labels(index),'.')
    hold on
end
xlabel('Column 2 of V')
ylabel('Column 3 of V')
zlabel('Column 5 of V')
legend({'0','1','2','3','4','5','6','7','8','9'});

col_v = [2,3,4,5,10,30,50];
acc = zeros(1,length(col_v));
for i=1:1:length(col_v)
    train_1 = proj(train_labels == 0, 2:col_v(i));
    train_2 = proj(train_labels == 4, 2:col_v(i));
    [res_1,y] = size(train_1);
    [res_2,y] = size(train_2);
    train = [train_1; train_2];
    adj_train = [0*ones(res_1,1); 4*ones(res_2,1)];

    test_t = (U'*test_data)';
    test_1 = test_t(test_labels == 0, 2:col_v(i));
    test_2 = test_t(test_labels == 4, 2:col_v(i));
    [res_1,y] = size(test_1);
    [res_2,y] = size(test_2);
    test = [test_1; test_2];
    adj_test = [0*ones(res_1,1); 4*ones(res_2,1)];

    class = classify(test, train, adj_train);

    err = sum(abs(adj_test - class)>0);
    acc(i)= 1-err/length(adj_test);
end
figure(4)
plot(col_v-1, acc, '-o')
xlabel('Number of Modes','Fontsize',12)
ylabel('Accuracy','Fontsize',12)


acc_lda = zeros(10,10);
for i=0:1:8
    for j=i+1:1:9
        train_1 = proj(train_labels == i,2:10);
        train_2 = proj(train_labels == j,2:10);
        [res_1,y] = size(train_1);
        [res_2,y] = size(train_2);
        train = [train_1; train_2];
        adj_train = [i*ones(res_1,1); j*ones(res_2,1)];

        test_t = (U'*test_data)';
        test_1 = test_t(test_labels==i,2:10);
        test_2 = test_t(test_labels==j,2:10);
        [res_1,y] = size(test_1);
        [res_2,y] = size(test_2);
        test = [test_1; test_2];
        adj_test = [i*ones(res_1,1); j*ones(res_2,1)];

        class = classify(test,train,adj_train);

        err = sum(abs(adj_test-class)>0);
        acc_lda(i+1,j+1)=1-err/length(adj_test);
    end
end

train_1 = proj(train_labels == 0,2:10);
train_2 = proj(train_labels == 4,2:10);
train_3 = proj(train_labels == 7,2:10);
[res_1,y] = size(train_1);
[res_2,y] = size(train_2);
[res_3,y] = size(train_3);
train=[train_1; train_2; train_3];
adj_train = [0*ones(res_1,1); 4*ones(res_2,1); 7*ones(res_3,1)];

test_t = (U'*test_data)';
test_1 = test_t(test_labels == 0,2:10);
test_2 = test_t(test_labels == 4,2:10);
test_3 = test_t(test_labels == 7,2:10);
[res_1,y] = size(test_1);
[res_2,y] = size(test_2);
[res_3,y] = size(test_3);
test = [test_1; test_2; test_3];
adj_test = [0*ones(res_1,1); 4*ones(res_2,1); 7*ones(res_3,1)];

class = classify(test,train,adj_train);

err = sum(abs(adj_test-class)>0);
accuracy_047 = 1-err/length(adj_test)

figure(5)
bar(class)
xlabel('Test Data')
ylabel('Prediction')


train = proj(:,2:10);
test_t = (U'*test_data)';
test = test_t(:,2:10);

class = classify(test,train,train_labels);

err = sum(abs(test_labels-class)>0);
acc_lda = 1-err/length(test_labels)


train = proj(:,2:10);
test_t = (U'*test_data)';
test = test_t(:,2:10);
Mdl = fitctree(train,train_labels,'OptimizeHyperparameters','auto');

class = predict(Mdl,test);

err = sum(abs(test_labels-class)>0);
accuracy_ct = 1-err/length(test_labels)


train = proj(:,2:10)/max(max(S));
test = test_t(:,2:10)/max(max(S));

SVM = cell(10,1);
group = 0:1:9;
rng(1);
for j = 1:numel(group)
    indx = train_labels==group(j);
    SVM{j} = fitcsvm(train,indx,'ClassNames',[false true],'Standardize',true,...
        'KernelFunction','rbf','BoxConstraint',1);
end
for j = 1:numel(group)
    [~,score] = predict(SVM{j},test);
    Scores(:,j) = score(:,2);
end

[~,maxi] = max(Scores,[],2);
err = sum(abs(test_labels+1-maxi)>0);
accuracy_svm = 1-err/length(test_labels)

digits = [4,9];

train_1 = proj(train_labels == digits(1),2:10);
train_2 = proj(train_labels == digits(2),2:10);
[res_1,y] = size(train_1);
[res_2,y] = size(train_2);
train = [train_1; train_2];
adj_train = [digits(1)*ones(res_1,1); digits(2)*ones(res_2,1)];

test_t = (U'*test_data)';
test_1 = test_t(test_labels == digits(1),2:10);
test_2 = test_t(test_labels == digits(2),2:10);
[res_1,y] = size(test_1);
[res_2,y] = size(test_2);
test = [test_1; test_2];
adj_test = [digits(1)*ones(res_1,1); digits(2)*ones(res_2,1)];

rng default
Md2 = fitcsvm(train,adj_train,'Standardize',true,...
        'KernelFunction','rbf','BoxConstraint',1);

class = predict(Md2,test);

err = sum(abs(adj_test-class)>0);
acc_svm_2 = 1-err/length(adj_test)

Md3 = fitctree(train,adj_train,'OptimizeHyperparameters','auto');

class = predict(Md3,test);

err = sum(abs(adj_test-class)>0);
accuracy_ct_2 = 1-err/length(adj_test)