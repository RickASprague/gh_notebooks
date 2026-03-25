Simple Julia LUX based autoencoder.  This is to just max IO.  Uses julia memory mapping.

* ImageNet 64 - 1.28M images
* 10 epochs

Done in the following steps:

1. Convert python input to mmap friendly flat files (1.28M images 64x64x3)
2. 10 epochs
3. Monitoring via Tensor Board
4. Lux.jl deep learning framework.

Setup
1. AMD Threadripper Pro 7965WX
   - 256 GB RAM
   - Ubuntu 24
   - ZFS - striped 2x4TB NVMe 5
2. RTX 6000 Blackwell Pro Worstation Edition

Trying to minimzing GPU memory footprint and minimize IO.

Notebooks
* Image Preprocess - converts image net 64 python pickles to memory mapped julia binary serialized files.
* Auto Encoder - trains the rudimentary autoencoder