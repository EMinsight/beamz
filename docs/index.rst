Welcome to BeamZ's documentation!
================================

BeamZ is a powerful tool for creating inverse designs for your photonic devices with ease and efficiency.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   api
   examples
   development
   license
   changelog

Installation
-----------

You can install BeamZ using pip:

.. code-block:: bash

   pip install beamz

For development installation, see the :doc:`development` guide.

Quick Start
----------

Here's a quick example of how to use BeamZ:

.. code-block:: python

   from beamz import FDTD, MaterialLibrary

   # Create a new FDTD simulation
   fdtd = FDTD()
   
   # Add materials
   materials = MaterialLibrary()
   materials.add_silicon()
   
   # Run simulation
   results = fdtd.run()

For more detailed examples, see the :doc:`examples` section.

Features
--------

* GPU-accelerated FDTD simulations
* Multi-threaded CPU backend
* Local execution for IP security
* Easy-to-use API
* Comprehensive material library

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 