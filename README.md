# Internal Redundant Coordinates

[![Build Status](https://travis-ci.org/RMeli/irc.svg?branch=master)](https://travis-ci.org/RMeli/irc)
[![codecov](https://codecov.io/gh/RMeli/irc/branch/master/graph/badge.svg)](https://codecov.io/gh/RMeli/irc)
[![License: MIT](https://img.shields.io/packagist/l/doctrine/orm.svg)](https://opensource.org/licenses/MIT)

## Usage

## Theory

### Internal Redundant Coordinates

## Contributions

Any contribution to this open source project is very welcome. If you are considering contributing you may find beneficial to have a look at the [Open Source Guides](https://opensource.guide/).

List of contributors:

- Rocco Meli (University of Bristol)
- Peter Bygrave (University of Bristol)

## Sources

### Internal Redoundant Coordinates

- P. Puly and G. Fogarasi, *Geometry optimization in redundant internal coordinates*, J. Chem. Phys. **96** 2856 (1992).

- C. Peng, P. Y. Ayala and H. B. Schlegel, *Using Redundant Internal Coordinates to Optimize Equilibrium Geometries and Transition States*, J. Comp. Chem. **17**, 49-56 (1996).

- V. Bakken and T. Helgaker, *The efficient optimization of molecular geometries using redundant internal coordinates*, J. Chem. Phys. **117**, 9160 (2002).

- E. Bright Wilson Jr., J. C. Decius and P. C. Cross, *Molecular Vibrations: The Theory of Infrared and Raman Vibrational Spectra*, Dover Publications Inc. (2003).

## Test suite

### Catch2
The IRC library is tested using the multi-paradigm test framework [Catch2](https://github.com/catchorg/Catch2), included as a single header file.

### CTest
Tests are run using the CTest testing tool distributed as a part of CMake.

  make -j test
  
### Travis-CI
[![Build Status](https://travis-ci.org/RMeli/irc.svg?branch=master)](https://travis-ci.org/RMeli/irc)

Continuous integration (CI) is implemented using [Travis-CI](https://travis-ci.org/) The test suite is run for every commit on the branches `master` and `travis-ci` and at least once a day for the `master` branch.
