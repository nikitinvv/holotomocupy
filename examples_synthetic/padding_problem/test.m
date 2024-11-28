n = 512;
ne = n+n/4;
energy = 33.35;
wavelength = 1.24e-09/energy;
ndist = 4;
distances = [2.9432e-3,3.06911e-3,3.57247e-3,4.61673e-3];
magnification = 400;
detector_pixelsize = 3.03751e-6;
voxelsize = detector_pixelsize/magnification*2048/n 

% load psi, size(psi)=(512+128,512+128)==(n+n/4,(n+n/4))
load('psi.mat');

% calculate data for 4 distances
data = zeros(n, n, ndist, 'like', psi);  
for i = 1:ndist
    % Fresnel kernel
    fx = fftshift((-ne:ne-1) / (2 * ne * voxelsize));
    [fx, fy] = meshgrid(fx, fx);  
    fP = exp(-1j * pi * wavelength * distances(i) * (fx.^2 + fy.^2));

    % pad psi
    ppsi = zeros(2*ne, 2*ne, 'like', psi);  
    ppsi(ne/2+1:3*ne/2,ne/2+1:3*ne/2) = psi;

    % convolution
    ppsi = ifft2(fft2(ppsi) .* fP);

    % crop
    data(:,:, i) = ppsi((ne - n/2 + 1):(ne + n/2), (ne - n/2 + 1):(ne + n/2));
end
