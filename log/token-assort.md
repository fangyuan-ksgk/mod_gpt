### Token Assorted (Baseline)
1. Build abstraction token labeled based on chunk of NL tokens, via VQ-VAE. 
* We found severe vocabulary collapse when training VQ-VAE with similar config, on GSM8K dataset, perhaps training on the mixed dataset reported in the TokenAssorted paper can avoid the vocabulary collapse, but we'll simplify the VQ-VAE into direct VQ on model's hidden representation, following our v6 VQ trick. 

2. Random abstraction token replacement (with <abs start> ... <abs end> deliminator). 
