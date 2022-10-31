# Quantum GAN for Small Molecule Drug

This library refers to the following source code.
* [yongqyu/MolGAN-pytorch](https://github.com/yongqyu/MolGAN-pytorch)


## Dependencies

* **python>=3.7**
* **pytorch>=0.4.1**: https://pytorch.org
* **rdkit**: https://www.rdkit.org
* **pennylane**
* **tensorflow==1.15**

## Dataset
* Run bash script `data/gdb9_generater.sh` to download gdb database and then run `data/sparse_molecular_dataset.py` to generate molecular graph dataset used to train the model.

## Training
```
python main.py --mode=train

```

## Prediction
To run the model against test dataset, make sure the model is fully trainned in the first place.
```
python main.py --mode=test
```
## Structure
`main.py` parse the command line arguments and pass it to the `Qgans_molGen.py` which access generator and discriminator model from `models.py` which inturn access `layers.py` and `utils.py` evaluate the metrics.  

## Generated Molecules
Below are some generated molecules:
![Alt text](images/generated_sample.png?raw=true "Generated Molecules")



