# opticalfibreml
Data and code for the paper "Deep, complex, invertible networks for inversion of transmission effects in multimode optical fibres" published at NIPS 2018

This repository accompanies the paper "Deep, complex, invertible networks for inversion of transmission effects in multimode optical fibres", Oisín Moran · Piergiorgio Caramazza · Daniele Faccio · Roderick Murray-Smith, published at NIPS 2018, Montreal Canada.

We use complex-weighted, deep convolutional networks to invert the effects of multimode optical fibre distortion of a coherent input image. We generated experimental data based on collections of optical fibre responses to greyscale, input images generated with coherent light, and measuring only image amplitude (not amplitude and phase as is typical) at the output of the 10 metre long 105 micrometre diameter multimode fibre. This data is made available as the *Optical fibre inverse problem Benchmark* collection. The experimental data is used to train complex-weighted models with a range of regularisation approaches and subsequent denoising autoencoders. A new *unitary regularisation* approach for complex-weighted networks is proposed which performs best in robustly inverting the fibre transmission matrix, and which fits well with the physical theory. The use of unitary layers allows analytic inversion of the network via its complex conjugate transpose, and we demonstrate simultaneous optimisation of both the forward and inverse models.

It includes the data used, and code to reproduce the images and results in the paper.
