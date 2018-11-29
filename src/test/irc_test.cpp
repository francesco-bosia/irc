#include "catch.hpp"

#include "config.h"

#include "libirc/irc.h"

#include "libirc/conversion.h"
#include "libirc/io.h"
#include "libirc/molecule.h"

#ifdef HAVE_ARMA
#include <armadillo>
using vec3 = arma::vec3;
using vec = arma::vec;
using mat = arma::mat;

template<typename T>
using Mat = arma::Mat<T>;
#elif HAVE_EIGEN3
#include <Eigen3/Eigen/Dense>
using vec3 = Eigen::Vector3d;
using vec = Eigen::VectorXd;
using mat = Eigen::MatrixXd;

template<typename T>
using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
#else
#error
#endif

using namespace irc;

TEST_CASE("Internal Redundant Coordinates") {
  using namespace std;

  using namespace connectivity;
  using namespace molecule;
  using namespace tools::conversion;

  SECTION("User-defined Coordinates") {

    // Define formaldehyde molecule (CH2O)
    Molecule<vec3> molecule{{"C", {0.000000, 0.000000, -0.537500}},
                            {"O", {0.000000, 0.000000, 0.662500}},
                            {"H", {0.000000, 0.866025, -1.037500}},
                            {"H", {0.000000, -0.866025, -1.037500}}};

    // Transform molecular coordinates from angstrom to bohr
    multiply_positions(molecule, angstrom_to_bohr);

    // Build internal reaction coordinates
    // Add O-H bonds: (1,2)
    // Add H-H bond: (2,3)
    // Add H-O-H angle: (2,1,3)
    // Add H-H-O-C dihedral: (3,2,1,0)
    IRC<vec3, vec, mat> irc(
        molecule, {{1, 2}, {2, 3}}, {{2, 1, 3}}, {{3, 2, 1, 0}}, {});
    REQUIRE(irc.get_bonds().size() == 5);
    REQUIRE(irc.get_angles().size() == 4);
    REQUIRE(irc.get_dihedrals().size() == 1);
    REQUIRE(irc.get_out_of_plane_bends().size() == 1);

    // Compute internal coordinates
    vec q_irc{
        irc.cartesian_to_irc(molecule::to_cartesian<vec3, vec>(molecule))};

    // Check size (3+2 bonds, 3+1 angles, 0+1 dihedrals)
    REQUIRE(linalg::size(q_irc) == 11);

    // Check manually added O-H bond
    SECTION("Manually added O-H bond") {
      Approx target(q_irc(3));
      target.margin(1e-6);

      // Compute O-H distance
      double d{distance(molecule[1].position, molecule[2].position)};

      REQUIRE(d == target);
    }

    // Check manually added H-H bond
    SECTION("Manually added H-H bond") {
      Approx target(q_irc(4));
      target.margin(1e-6);

      // Compute H-H disance
      double b{distance(molecule[2].position, molecule[3].position)};

      REQUIRE(b == target);
    }

    // Check manually added H-O-H angle
    SECTION("Manually added H-O-H angle") {
      Approx target(q_irc(8));
      target.margin(1e-6);

      // Compute H-O-H angle
      double a{angle(
          molecule[2].position, molecule[1].position, molecule[3].position)};

      REQUIRE(a == target);
    }

    // Check manually added H-H-O-C dihedral angle
    SECTION("Manually added H-H-O-C dihedral angle") {
      Approx target(q_irc(9));
      target.margin(1e-6);

      // Compute H-H-O-C dihedral angle
      double d{dihedral(molecule[3].position,
                        molecule[2].position,
                        molecule[1].position,
                        molecule[0].position)};

      REQUIRE(d == target);
    }

    // Check manually added H-H-O-C dihedral angle
    SECTION("Manually added H-H-O-C out of plane angle") {
      Approx target(q_irc(10));
      target.margin(1e-6);

      // Compute H-H-O-C dihedral angle
      double d{out_of_plane_angle(molecule[0].position,
                                  molecule[1].position,
                                  molecule[2].position,
                                  molecule[3].position)};

      REQUIRE(d == target);
      REQUIRE(d == Approx(0).margin(1e-6));
    }
  }
  /*
  SECTION("Constraints") {
    // Define formaldehyde molecule (CH2O)
    Molecule<vec3> molecule{{"C", {0.000000, 0.000000, -0.537500}},
                            {"O", {0.000000, 0.000000, 0.662500}},
                            {"H", {0.000000, 0.866025, -1.037500}},
                            {"H", {0.000000, -0.866025, -1.037500}}};
    
    // Transform molecular coordinates from angstrom to bohr
    multiply_positions(molecule, angstrom_to_bohr);
    
    // Build internal reaction coordinates
    // Add C-O bond constraint: (0,1)
    // Add H-H bond constraint: (2,3)
    // Add H-C-H angle constraint: (2,0,3)
    // Add H-H-O-C dihedral constraint: (3,2,1,0)
    IRC<vec3, vec, mat> irc(
        molecule, {{0,1,Constraint::constrained}, {2,3,Constraint::constrained}}, {{2,0,3,Constraint::constrained}}, {{3,2,1,0,Constraint::constrained}});
    
    // Compute internal coordinates
    vec q_irc{
        irc.cartesian_to_irc(molecule::to_cartesian<vec3, vec>(molecule))};
    
    // Check size (3+1 bonds, 3+0 angles, 0+1 dihedrals)
    REQUIRE(linalg::size(q_irc) == 8);
    
    // Get bonds
    auto B = irc.get_bonds();
    
    // Check bonds
    REQUIRE(B.size() == 4 );
    CHECK(B[0].constraint == Constraint::constrained);
    CHECK(B[3].constraint == Constraint::constrained);
  
    // Get angles
    auto A = irc.get_angles();
  
    // Check bonds
    REQUIRE(A.size() == 3 );
    CHECK(A[2].constraint == Constraint::constrained);
  
    // Get angles
    //auto D = irc.get_dihedrals();
  
    // Check bonds
    //REQUIRE(D.size() == 1 );
    CHECK(D[0].constraint == Constraint::constrained);
  }
  */

  SECTION("Initial hessian") {

    // Define formaldehyde molecule (CH2O)
    Molecule<vec3> molecule{{"C", {0.000000, 0.000000, -0.537500}},
                            {"O", {0.000000, 0.000000, 0.662500}},
                            {"H", {0.000000, 0.866025, -1.037500}},
                            {"H", {0.000000, -0.866025, -1.037500}}};

    // Transform molecular coordinates from angstrom to bohr
    multiply_positions(molecule, angstrom_to_bohr);

    // Build internal reaction coordinates
    // (Manually added dihedral to increase coverage)
    IRC<vec3, vec, mat> irc(molecule, {}, {}, {{0, 1, 2, 3}});

    // Compute initial hessian
    mat iH0{irc.projected_initial_hessian_inv()};

    // Project Hessian again
    mat iH{irc.projected_hessian_inv(iH0)};

    // Check sizes
    REQUIRE(linalg::size(iH0) == linalg::size(iH));

    // Check that second projection has no effect
    std::size_t n{linalg::size(iH0)};
    for (std::size_t i{0}; i < n; i++) {
      Approx target(iH0(i));
      target.margin(1e-6);

      REQUIRE(iH(i) == target);
    }
  }

  SECTION("IRC to Cartesian") {

    // Define formaldehyde molecule (CH2O)
    const auto molecule =
        io::load_xyz<vec3>(config::molecules_dir + "ethanol.xyz");

    // Build internal reaction coordinates
    IRC<vec3, vec, mat> irc(molecule);

    // Get cartesian coordinates
    vec x_c{to_cartesian<vec3, vec>(molecule)};

    // Compute internal redundant coordinates
    vec q_irc{irc.cartesian_to_irc(x_c)};

    // Define no displacement in IRC
    vec dq{linalg::zeros<vec>(linalg::size(q_irc))};

    // Compute cartesian coordinate from IRC
    vec x_c_from_irc{irc.irc_to_cartesian(q_irc, dq, x_c)};

    std::size_t n{linalg::size(x_c)};
    for (std::size_t i{0}; i < n; i++) {
      Approx target(x_c(i));
      target.margin(1e-6);

      REQUIRE(x_c_from_irc(i) == target);
    }
  }
}
