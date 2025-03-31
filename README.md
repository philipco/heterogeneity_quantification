# heterogeneity_quantification

## GPU usage

If the GPU is not find:
1. ```sudo lsof /dev/nvidia-uvm```: identify moduls that use the GPU
2. Kill them.
3. ```sudo rmmod nvidia_uvm```
4. ```sudo modprobe nvidia_uvm```
