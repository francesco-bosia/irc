#define CATCH_CONFIG_MAIN
#include "../catch/catch.hpp"

#include "molecule.h"

#include "../atoms/periodic_table.h"
#include "../tools/comparison.h"

#include <cassert>
#include <iostream>

#ifdef HAVE_ARMA
#include <armadillo>
using vec3 = arma::vec3;
#else
#error
#endif

TEST_CASE("Molecule") {
  
  using namespace std;
  using namespace molecule;
  using namespace periodic_table;
  using namespace tools::comparison;
  
  Molecule<vec3> molecule{
      {1, {0.0, 1.1, 2.2}},
      {2, {0.0, 1.1, 2.2}},
      {3, {0.0, 1.1, 2.2}}
  };
  
  SECTION("Mass") {
    Approx target{pt_masses[1] + pt_masses[2] + pt_masses[3]};
    
    target.margin(1e-12);
    
    REQUIRE( mass(molecule) == target );
  }
  
  SECTION("Positon multiplier") {
    multiply_positions(molecule, 2.);
    
    for (const auto &atom : molecule) {
      REQUIRE( nearly_equal(atom.position, vec3{0.0, 2.2, 4.4}) );
    }
  }
  
}