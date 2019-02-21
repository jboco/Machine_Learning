clear all;
close all;
load ('data.mat');
%reshape data into 504*600 matrix
face2= reshape(face,21*24,600);
min2=min(min(face2));
max2=max(max(face2));
%normalize data 
face3=(((face2-min2.*ones(504,600)))./(max2-min2));
q=1;
k=1;
training_data_mean=zeros(size(face3, 1),1);
indexer=0;
%find mean of training data
while q<599  
  
    switch indexer   
      
    case 0  
        q=q+1;
        
       training_data_mean= training_data_mean + face3(:,q)+ face3(:,q-1); 
        
    case 1  
        q=q+2;  
       
    end  
    indexer=indexer+1;  
    if indexer > 1  
        indexer = 0;     
    end
end 
training_data_mean=training_data_mean./400;
%center data
centered_face3=zeros(504,600);
for i=1:600
centered_face3(:,i)= face3(:,i)-training_data_mean;
end
i= 1; 
k=1;
switcher = 0;
Sw=zeros(504, 504);
Sb=zeros(504, 504);
Si=zeros(504,504,200);
%find mean of all samples
total_mean=zeros(504,1);
 for i=1:600
     total_mean=total_mean+face3(:,i);
 end
 total_mean=total_mean./600;
while i<599  
  
    switch switcher   
      
    case 0  
        i=i+1;
        
       mu_i(:,k)=(face3(:,i)+ face3(:,i-1))/2; 
       Si(:,:,k)= (face3(:,i)-mu_i(:,k))*transpose(face3(:,i)-mu_i(:,k))+(face3(:,i-1)-mu_i(:,k))*transpose(face3(:,i-1)-mu_i(:,k)); 
       Sw= Sw+Si(:,:,k);
       Sb=Sb+(2*(mu_i(:,k)-training_data_mean)*transpose(mu_i(:,k)-training_data_mean));
       k=k+1;
    case 1  
        i=i+2;  
       
    end  
    switcher=switcher+1;  
    if switcher > 1  
        switcher = 0;     
    end 
end 
%find eigenvectors
[V, L]=eig(Sb,Sw);                  % calc e-vectors (V) & e-values (L) 
[L, iL]=sort(diag(L),'descend'); % sort L in descending
L=diag(L);                       % reform eigenvalue matrix
V=V(:,iL); 
%project data on feature vectors
transformed_data= V([1:199],:)*centered_face3;
indexer=0;
q=1;
k=1;
m=input('enter test sample column index (3rd image of each class): ' );
%find 1st nearest neighbors
while q<599  
  
    switch indexer   
      
    case 0  
        q=q+1;
        
       distance(k)= min(norm((transformed_data(:,m)-transformed_data(:,q)),1), norm((transformed_data(:,m)-transformed_data(:,q-1)), 1)); 
        k=k+1;
    case 1  
        q=q+2;  
       
    end  
    indexer=indexer+1;  
    if indexer > 1  
        indexer = 0;     
    end    
end 
[B,Id]=min(distance);
fprintf('The sample belongs to class %d\n',Id);
