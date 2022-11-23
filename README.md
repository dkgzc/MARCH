# MARCH

The repo is for MARCH: Communication-Aware Heterogeneous Accelerator Mapping Method for DNN Model\

It consists of three parts: Accelerator, Modality, and Mapping.\
You can read and modify the accelerator/Modality configuration in Accelerator/Modality folder. Customized accelerator/modality design is enabled.\

Mapping:\
Optimal.py in Mapping is the main algorithm. Run it for a high quality Multi-Modality Multi-Accelerator (MMMA) mapping result. Default configuration is VlocNet onto given accelerator configuration.\
Optimal_multi.py is the algorithm for cluster configuration with multilevel communication
