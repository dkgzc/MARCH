# MARCH

The repo is for MARCH: Communication-Aware Heterogeneous Accelerator Mapping Method for DNN Model

It consists of three parts: accelerator, modality, and mapper.\
You can read and modify the accelerator/modality configuration in accelerator/modality folder. Customized accelerator/modality design is enabled.

mapper:\
Optimal.py in mapper folder is the main algorithm. Run it for a high quality Multi-Modality Multi-Accelerator (MMMA) mapping result. Default configuration is VlocNet onto given accelerator configuration.\
Optimal_multilevel.py is the algorithm for cluster configuration with multilevel communication. Default configuration is case 1 mentioned in paper.

Try to simply run Optimal.py/Optimal_multilevel.py to get a default result.\
You can change the 'AccsLists' or 'Modalities' to get the all results in this paper.
