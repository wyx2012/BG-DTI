# BG-DTI
BG-DTI: A biological feature and heterogeneous network representation learning-based framework for drug–target interaction prediction

###Citation：
If you have used BG-DTI or its modules in your research, please cite this paper

## 1. Environment setup
  python:3.7.6  

  torch:1.4.0  

  scikit-learn:0.22.1  

  pandas:1.0.1  

  numpy:1.18.1

##2.Construct a dataset
  if you want using other datasets, you can proceed as follows:
  1.Similarity calculation
  drug_rdkit_SimMat：Similarity calculation of medicinal chemistry formula (where rdkit needs to be run in python 3.9)
  pertion_fasta_SimMat：Target sequence similarity calculation

  2.Edge Computing
  z_add_edge:Generate edge matrix and store it in CSV format

##3.start:
  python main.py  # BG-DTI
