
# Classifying Breast Cancer Molecular Subtypes Using Deep Clustering Approach

Cancer is a complex disease with a high rate of mortality. The characteristics of tumor masses are very heterogeneous; thus, the appropriate classification of tumors is a critical point in the correct treatment. A high level of heterogeneity has also been observed in breast cancer. Therefore, detecting the molecular subtypes of this disease is an essential issue for medicine that could be facilitated using bioinformatics.
This repository contains the implementation codes of the MSDEC, a method for molecular subtype detection. The paper is available in https://www.frontiersin.org/articles/10.3389/fgene.2020.553587/abstract
## MSDEC schema

![MSDEC schema](https://github.com/nrohani/MolecularSubtypes/blob/master/MSDEC%20Schema.jpg)

### Prerequisites and Installing packages

1. Install Python 3.x
2. Install Keras>=2.0.9
3. Install scikit-learn 
```
pip install keras
pip install scikit-learn   
```

### Materials

Data: We used breast cancer somatic mutation profiles collected by [Zhang et al.](https://github.com/wzhang1984/NBSS/tree/master/data). They have obtained 78 somatic mutation data of 861 breast tumors from TCGA. You can also find this data in [Data.txt](https://github.com/nrohani/MolecularSubtypes/blob/master/Data/Data.txt). We propagated this profile using PPI network ([PPI.txt](https://github.com/nrohani/MolecularSubtypes/blob/master/Data/PPI.txt)) and obtained [propagatedData.txt](https://github.com/nrohani/MolecularSubtypes/blob/master/Data/propagatedData.txt).

### Codes
Find MSDEC implementation in [MSDEC.py](https://github.com/nrohani/MolecularSubtypes/blob/master/MSDEC.py)
### Results
The result's folder contains the result of finding the best number of clusters based on AUMP and other metrics.
## Authors

* **Narjes Rohani** and **Changiz Eslahchi**
Please do not hesitate to contact us if you have any question:


Mail: narjesrohani1996@gmail.com
