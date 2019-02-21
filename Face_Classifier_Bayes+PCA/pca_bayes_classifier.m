clear all;
close all;
load ('data.mat');

%reshape data into 504*600 matrix
face2= reshape(face,21*24,600);
min2=min(min(face2));
max2=max(max(face2));

%normalize data
face3=(((face2-min2.*ones(504,600)))./(max2-min2));
%center data around mean
centered_face3=zeros(504,600);
q=1;
tot_mean=zeros(size(face3, 1),1);
indexer=0;
while q<599  
  
    switch indexer   
      
    case 0  
        q=q+1;
        
       tot_mean= tot_mean + face3(:,q)+ face3(:,q-1); 
        
    case 1  
        q=q+2;  
       
    end  
    indexer=indexer+1;  
    if indexer > 1  
        indexer = 0;     
    end
    

end 
tot_mean=tot_mean./400;

for i=1:600
centered_face3(:,i)= face3(:,i)-tot_mean;
end

C= centered_face3 * transpose (centered_face3);
[V, L]=eig(C);                  % calc e-vectors (V) & e-values (L) 
[L, iL]=sort(diag(L),'descend'); % sort L in descending
L=diag(L);                       % reform eigenvalue matrix
V=V(:,iL);  % reorder eigenvectors
size (V);
feature_vec= V([1:400],:); %select first 400 eigenvectors
size(feature_vec);
reduced_data=  feature_vec * centered_face3; %project centered data on eigenspace
size(reduced_data);
 i= 1; 
 k=1;
 switcher = 0;
 avg_x=zeros(size(reduced_data,1),200);
 sigma=zeros(size(reduced_data,1), size(reduced_data,1), 200); 
 
while i<599  
  
    switch switcher   
      
    case 0  
        i=i+1;
       %calculate mean and variance of each class 
       avg_x(:,k)=(reduced_data(:,i)+ reduced_data(:,i-1))/2; 
        sigma(:,:,k)=(((reduced_data(:,i) - avg_x(:,k))* transpose(reduced_data(:,i) - avg_x(:,k))) + ((reduced_data(:,i-1) - avg_x(:,k))* transpose(reduced_data(:,i-1) - avg_x(:,k))))/2 +1*eye(size(reduced_data,1),size(reduced_data,1));

       k=k+1;
    case 1  
        i=i+2;  
       
    end  
    switcher=switcher+1;  
    if switcher > 1  
        switcher = 0;     
    end 
     

end 
m=input('enter test sample column index (3rd image of each class): ' );
%classify sample
for n=1:200
 result(n)= (1/((det(sigma(:,:,n)))^0.5))*(exp ( -0.5 .* transpose( reduced_data(:,m) - avg_x(:,n))* pinv(sigma(:,:,n))*( reduced_data(:,m) - avg_x(:,n))));
end
[A,I]=max(result);
fprintf('The sample belongs to class %d\n',I);
