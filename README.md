SPProC: Sequential learning with Physical Probabilistic Constraints
===========
## Description

Physics informed batch Bayesian optimization for optimizing perovskite stability. Codes and data are described in the connecting article:

Shijing Sun, Armi Tiihonen, Felipe Oviedo, Zhe Liu, Janak Thapa, Noor Titan P. Hartono, Anuj Gopal, Clio Batali, Alex Encinas, Jason J. Yoo, Ruipeng Li, Zekun Ren, Moungi G. Bawendi, Vladan Stevanovic, John Fisher III, Tonio Buonassisi, "A Physical Data Fusion Approach to Optimize Compositional Stability of Halide Perovskites" (2020), link: https://chemrxiv.org/articles/preprint/A_Physical_Data_Fusion_Approach_to_Optimize_Compositional_Stability_of_Halide_Perovskites/12601997

## Installation
To install, just clone the following repository and sub-repository:

`$ git clone https://github.com/PV-Lab/SPProC.git`

`$ cd SPProC`

`$ cd GPyOpt_DFT`

`$ git clone https://github.com/PV-Lab/GPyOpt_DFT.git`

To install the modified GPyOpt package, create a virtual environment using Anaconda (Optional but recommended setup):

`$ conda create --name SPProC python=3.7`

`$ conda activate SPProC`

`$ conda install spyder`

Run the following terminal commands to setup the package:

`$ python setup.py install`

`$ pip install -r requirements.txt`

Open spyder and run SPProC/Main.py

## Authors
||                    |
| ------------- | ------------------------------ |
| **AUTHORS**      | Armi Tiihonen, Felipe Oviedo, Shreyaa Raghavan, Zhe Liu     | 
| **VERSION**      | 1.0 / June, 2020     | 
| **EMAILS**      | armi.tiihonen@gmail.com, foviedo@mit.edu, shreyaar@mit.edu, chris.liuzhe@gmail.com  | 
||                    |

## Attribution
This work is under an Apache 2.0 License. Please, acknowledge use of this work with the appropiate citation to the repository and research article.

## Citation

    @Misc{spproc2020,
      author =   {The SPProC authors},
      title =    {{SPProC}: Sequential learning with Physical Probabilistic Constraints},
      howpublished = {\url{https://github.com/PV-Lab/SPProC}},
      year = {2020}
    }
    
    {To be added: citation of the article}
