
# Classifying Breast Cancer Molecular Subtypes Using Deep Clustering Approach

Cancer is a complex disease with a high rate of mortality. The characteristics of tumor masses are very heterogeneous; thus, the appropriate classification of tumors is a critical point in the correct treatment. A high level of heterogeneity has also been observed in breast cancer. Therefore, detecting the molecular subtypes of this disease is a worthwhile issue for medicine that could be facilitated using bioinformatics.
This repository contains the implementated codes of the MSDEC, a method for molecular subtype detection. Preprint version is avialable in https://www.researchsquare.com/article/rs-10143/v1

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
Find DEC clustering implementation in DEC.py
### Results folder



