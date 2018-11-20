img = load('data.mat');
H1 = img.H1;
H2 = img.H2;
r = 1.5;
%{
subplot(1,2,1)
imshow(H1,[]);
subplot(1,2,2)
imshow(H2,[]);
%}

ind1 = H2 < 0;
ind2 = (1 + tanh(H2) < r);
imshow(ind1.*ind2,[]);