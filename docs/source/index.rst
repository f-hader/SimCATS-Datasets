.. SimCATS documentation master file, created by
   sphinx-quickstart on Wed Oct 18 10:23:06 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. include:: misc/README.md
   :parser: myst_parser.sphinx_
   :end-before: <!-- start sec:documentation -->

.. include:: misc/README.md
   :parser: myst_parser.sphinx_
   :start-after: <!-- end sec:documentation -->

.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Examples

   notebooks/*

.. toctree::
   :maxdepth: 3
   :caption: API Reference

   /autoapi/simcats_datasets/index
