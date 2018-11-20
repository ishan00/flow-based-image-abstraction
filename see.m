img = load('macaw.mat');
H1 = img.H1;
H2 = img.H2;
tang = img.tang;
r = 0.999;


figure(1);
subplot(2,2,1)
imshow(tang(:,:,1),[]);
subplot(2,2,2)
imshow(tang(:,:,2),[]);
subplot(2,2,3)
imshow(H1,[]);
subplot(2,2,4)
imshow(H2,[]);

figure(2);
ind1 = H2 < 0;
ind2 = ((1 + tanh(H2)) < r);

imshow(ind1.*ind2,[]);