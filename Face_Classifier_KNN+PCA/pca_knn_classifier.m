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

%center data
centered_face3=zeros(504,600);
for i=1:600
centered_face3(:,i)= face3(:,i)-tot_mean;
end

%find eigenvectors
C= centered_face3 * transpose (centered_face3);
[V, L]=eig(C);                  % calc e-vectors (V) & e-values (L) 
[L, iL]=sort(diag(L),'descend'); % sort L in descending
L=diag(L);                       % reform eigenvalue matrix
V=V(:,iL);  % reorder eigenvectors
size (V);
%select first 400 eigenvectors 
feature_vec= V([1:400],:);
size(feature_vec);
%project data on eigensapce
reduced_data=  feature_vec * centered_face3;

m=input('enter test sample column index (3rd image of each class): ' );

%find 1st nearest neighbor 
indexer=0;
q=1;
k=1;
while q<599  
  
    switch indexer   
      
    case 0  
        q=q+1;
        
       dist(k)= min(norm((reduced_data(:,m)-reduced_data(:,q)),1), norm((reduced_data(:,m)-reduced_data(:,q-1)), 1)); 
        k=k+1;
    case 1  
        q=q+2;  
       
    end  
    indexer=indexer+1;  
    if indexer > 1  
        indexer = 0;     
    end

end 
[A,I]=min(dist);
fprintf('The sample belongs to class %d\n',I);
