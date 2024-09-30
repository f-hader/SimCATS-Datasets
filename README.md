<h1 align="center">
  <img src="https://raw.githubusercontent.com/f-hader/SimCATS-Datasets/main/SimCATS-Datasets_symbol.svg" alt="SimCATS logo">
  <br>
</h1>

<div align="center">
  <a href="https://github.com/f-hader/SimCATS-Datasets/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-GPLv3-blue.svg" alt="License: GPLv3"/>
  </a>
  <a href="https://pypi.org/project/simcats-datasets/">
    <img src="https://img.shields.io/pypi/v/simcats-datasets.svg" alt="PyPi Latest Release"/>
  </a>
  <a href="https://simcats-datasets.readthedocs.io/en/latest/">
    <img src="https://img.shields.io/readthedocs/simcats-datasets" alt="Read the Docs"/>
  </a>
  <a href="https://doi.org/10.1109/TQE.2024.3445967">
    <img src="https://img.shields.io/badge/DOI (SimCATS Paper)-10.1109/TQE.2024.3445967-007ec6.svg" alt="DOI Paper"/>
  </a>
  <a href="https://doi.org/10.5281/zenodo.13862231">
    <img src="https://img.shields.io/badge/DOI (Code)-10.5281/zenodo.13862231-007ec6.svg" alt="DOI Code"/>
  </a>
</div>

# SimCATS-Datasets

`SimCATS-Datasets` is a Python package that simplifies the creation and loading of `SimCATS` datasets. Please have a look at 
[this repository](https://github.com/f-hader/SimCATS) regarding `SimCATS` itself.

## Installation

The framework supports Python versions 3.7 - 3.11 and installs via pip:
```
pip install simcats-datasets
```

Alternatively, the `SimCATS-Datasets` package can be installed by cloning the GitHub repository, navigating to the
folder containing the `setup.py` file, and executing
```
pip install .
```

For installation in development/editable mode, use the option `-e`.

<!-- start sec:documentation -->
## Documentation

The official documentation is hosted on [ReadtheDocs](https://simcats-datasets.readthedocs.io) but can also be built
locally. To do this, first install the packages `sphinx`, `sphinx-rtd-theme`, `sphinx-autoapi`, `myst-nb `, and 
`jupytext` with

```
pip install sphinx sphinx-rtd-theme sphinx-autoapi myst-nb jupytext
```

and then, in the `docs` folder, execute the following command:

```
.\make html
```

To view the generated HTML documentation, open the file `docs\build\html\index.html`.
<!-- end sec:documentation -->


## Loading Datasets

Datasets created with `SimCATS-Datasets` are stored in HDF5 files. These datasets can be loaded using the function 
`load_dataset` from `simcats_datasets.loading`.

The return value of the function is a named tuple. The fields can be accessed by their name or index. As with normal
tuples, it is also possible to unpack the returned fields directly into separate variables. The available fields 
depend on which data was specified to be loaded. Please look at the docstring for further information.

Additionally, `SimCATS-Datasets` offers a pytorch dataset (see `torch.utils.data.Dataset`) implementation called 
`SimcatsDataset`. It allows the direct use of `SimCATS` datasets for machine learning purposes with Torch and can be 
imported from `simcats_datasets.loading.pytorch`.

## Creating Datasets

To create a simulated dataset, import `create_simulated_dataset` from `simcats_datasets.generation`. This function
allows the creation of simulated CSDs with ground truth very easily. It is also possible to add further CSDs to already
existing datasets. The function will detect the existing dataset automatically. For the function's usage, please have a
look at its docstring. 

| :warning: WARNING                                                                                                                                                                                                                                                                                                        |
|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| The functionalities for creating and extending simulated datasets using SimCATS expect that the SimCATS simulation uses the IdealCSDInterface implementation called IdealCSDGeometric. Other implementations might cause problems because the expected information for creating labeled lines etc. might be unavailable. |  


Alternatively, to using `create_simulated_dataset` and directly simulating a dataset with `SimCATS`, it is also possible 
to create a `SimCATS-Dataset` compatible dataset with existing data (for example, experimentally measured data or data 
simulated with other frameworks). This can be done using `create_dataset` from `simcats_datasets.generation`.

## Citations

```bibtex
@article{hader2024simcats,
  author={Hader, Fabian and Fleitmann, Sarah and Vogelbruch, Jan and Geck, Lotte and Waasen, Stefan van},
  journal={IEEE Transactions on Quantum Engineering}, 
  title={Simulation of Charge Stability Diagrams for Automated Tuning Solutions (SimCATS)}, 
  year={2024},
  volume={5},
  pages={1-14},
  doi={10.1109/TQE.2024.3445967}
}
```

## License, CLA, and Copyright

[![CC BY-NC-SA 4.0][gplv3-shield]][gplv3]

This work is licensed under a
[GNU General Public License 3][gplv3].

[![GPLv3][gplv3-image]][gplv3]

[gplv3]: https://www.gnu.org/licenses/gpl-3.0.html
[gplv3-image]: https://www.gnu.org/graphics/gplv3-127x51.png
[gplv3-shield]: https://img.shields.io/badge/License-GPLv3-blue.svg

Contributions must follow the Contributor License Agreement. For more information, see the [CONTRIBUTING.md](https://github.com/f-hader/SimCATS-Datasets/blob/main/CONTRIBUTING.md) file at the top of the GitHub repository.

Copyright © 2024 Forschungszentrum Jülich GmbH - Central Institute of Engineering, Electronics and Analytics (ZEA) - Electronic Systems (ZEA-2)
