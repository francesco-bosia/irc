#include "catch.hpp"

#include "libirc/connectivity.h"

#include "config.h"
#include "libirc/atom.h"
#include "libirc/conversion.h"
#include "libirc/io.h"
#include "libirc/molecule.h"
#include "libirc/transformation.h"

#include <iostream>

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

template<typename Vector3, typename Vector, typename Matrix>
std::tuple<std::vector<connectivity::Bond>,
           std::vector<connectivity::Angle>,
           std::vector<connectivity::Dihedral>>
bad_from_molecule(const molecule::Molecule<Vector3> &mol) {

  using namespace connectivity;

  // Compute interatomic distance for formaldehyde molecule
  Matrix dd{distances<Vector3, Matrix>(mol)};

  // Build graph based on the adjacency matrix
  UGraph adj{adjacency_matrix(dd, mol)};

  // Compute distance matrix and predecessor matrix
  Matrix dist, predecessors;
  std::tie(dist, predecessors) = distance_matrix<Matrix>(adj);

  // Compute bonds
  std::vector<Bond> B{bonds(dist, mol)};

  // Compute angles
  std::vector<Angle> A{angles(dist, predecessors, mol)};

  // Compute dihedral angles
  std::vector<Dihedral> D{dihedrals(dist, predecessors, mol)};

  // Return bonds, angles and dihedral angles
  return std::make_tuple(B, A, D);
}

TEST_CASE("Distance, angle and dihedral angle") {
  using namespace connectivity;

  // Define four points in 3D space
  const auto p1 = vec3{0.00, 0.00, -0.25};
  const auto p2 = vec3{0.00, 0.00, 1.50};
  const auto p3 = vec3{0.00, 1.00, 1.50};
  const auto p4 = vec3{5.00, 1.00, 1.50};

  REQUIRE(distance(p1, p2) == Approx(1.75));

  REQUIRE(angle(p1, p2, p3) == Approx(tools::constants::pi / 2.));

  REQUIRE(dihedral(p1, p2, p3, p4) == Approx(-tools::constants::pi / 2.));
}

// Regular bond
TEST_CASE("Connectivity for compressed H2") {
  using namespace std;

  using namespace tools::conversion;
  using namespace molecule;
  using namespace connectivity;

  double d{0.5};

  // Define compressed H2 molecule
  Molecule<vec3> molecule{{"H", {0.0, 0.0, d}}, {"H", {0.0, 0.0, 0.0}}};

  // Transform molecular coordinates from angstrom to bohr
  multiply_positions(molecule, angstrom_to_bohr);

  // Get bonds, angles and dihedrals
  std::vector<Bond> B;
  std::vector<Angle> A;
  std::vector<Dihedral> D;
  std::tie(B, A, D) = bad_from_molecule<vec3, vec, mat>(molecule);

  // Check number of bonds
  REQUIRE(B.size() == 1);

  // Check number of angles
  REQUIRE(A.size() == 0);

  // Check number of dihedral angles
  REQUIRE(D.size() == 0);

  SECTION("Bond") {
    // Compute IRC
    vec q{connectivity::cartesian_to_irc<vec3, vec>(
        to_cartesian<vec3, vec>(molecule), B, {}, {})};

    // Check number of IRC
    REQUIRE(linalg::size<vec>(q) == 1);

    // Check bond length
    Approx bb(d * angstrom_to_bohr);
    bb.margin(1e-12);
    REQUIRE(q(0) == bb);
  }
}

// Interfragment bond
TEST_CASE("Connectivity for stretched H2") {
  using namespace std;

  using namespace tools::conversion;
  using namespace molecule;
  using namespace connectivity;

  double d{2.5};

  // Define compressed H2 molecule
  Molecule<vec3> molecule{{"H", {0.0, 0.0, d}}, {"H", {0.0, 0.0, 0.0}}};

  // Transform molecular coordinates from angstrom to bohr
  multiply_positions(molecule, angstrom_to_bohr);

  // Get bonds, angles and dihedrals
  std::vector<Bond> B;
  std::vector<Angle> A;
  std::vector<Dihedral> D;
  std::tie(B, A, D) = bad_from_molecule<vec3, vec, mat>(molecule);

  // Check number of bonds
  REQUIRE(B.size() == 1);

  // Check number of angles
  REQUIRE(A.size() == 0);

  // Check number of dihedral angles
  REQUIRE(D.size() == 0);

  SECTION("Bond") {
    // Compute IRC
    vec q{connectivity::cartesian_to_irc<vec3, vec>(
        to_cartesian<vec3, vec>(molecule), B, {}, {})};

    // Check number of IRC
    REQUIRE(linalg::size<vec>(q) == 1);

    // Check bond length
    Approx bb(d * angstrom_to_bohr);
    bb.margin(1e-12);
    REQUIRE(q(0) == bb);
  }
}

// Angle
TEST_CASE("Connectivity for compressed H2O") {
  using namespace std;

  using namespace tools::conversion;
  using namespace molecule;
  using namespace connectivity;

  double d1{0.8};
  double d2{0.6};
  double angle{134};

  double angle_rad{angle * deg_to_rad};
  double a{(180. - angle) / 2.};
  double a_rad{a * deg_to_rad};

  // Define compressed H2 molecule
  Molecule<vec3> molecule{
      {"O", {0, 0, 0}},
      {"H", {d1 * std::cos(a_rad), -d1 * std::sin(a_rad), 0}},
      {"H", {-d2 * std::cos(a_rad), -d2 * std::sin(a_rad), 0}}};

  // Transform molecular coordinates from angstrom to bohr
  multiply_positions(molecule, angstrom_to_bohr);

  // Get bonds, angles and dihedrals
  std::vector<Bond> B;
  std::vector<Angle> A;
  std::vector<Dihedral> D;
  std::tie(B, A, D) = bad_from_molecule<vec3, vec, mat>(molecule);

  // Check number of bonds
  REQUIRE(B.size() == 2);

  // Check number of angles
  REQUIRE(A.size() == 1);

  // Check number of dihedral angles
  REQUIRE(D.size() == 0);

  // Compute IRC
  vec q{connectivity::cartesian_to_irc<vec3, vec>(
      to_cartesian<vec3, vec>(molecule), B, A, {})};

  // Check number of IRC
  REQUIRE(linalg::size<vec>(q) == 3);

  SECTION("Bonds") {
    Approx bb1(d1 * angstrom_to_bohr);
    bb1.margin(1e-12);
    REQUIRE(q(0) == bb1);

    Approx bb2(d2 * angstrom_to_bohr);
    bb2.margin(1e-12);
    REQUIRE(q(1) == bb2);
  }

  SECTION("Angle") {
    Approx aa(angle_rad);
    aa.margin(1e-12);
    REQUIRE(q(2) == aa);
  }
}

// Angle from interfragment bonds
TEST_CASE("Connectivity for stretched H2O") {}

// Dihedral
TEST_CASE("Connectivity for compressed H2O2") {
  // TODO
}

// Dihedral from interfragment bonds
TEST_CASE("Connectivity for stretched H2O2") {
  // TODO
}

// Hydrogen bond (without quasi-linear angles)
TEST_CASE("Connectivity for bent water dimer") {
  using namespace std;

  using namespace tools::conversion;
  using namespace molecule;
  using namespace connectivity;

  double d1{1.3};
  double d2{1.4};
  double angle{102.03};

  double a{(180. - angle) * deg_to_rad};

  double cos_a{std::cos(a)};
  double sin_a{std::sin(a)};

  // Define compressed H2 molecule
  Molecule<vec3> molecule{{"O", {-1.464, 0.099, -0.300}},
                          {"H", {-1.956, 0.624, -0.340}},
                          {"H", {-1.797, -0.799, 0.206}},
                          {"O", {1.369, 0.146, -0.395}},
                          {"H", {1.894, 0.486, 0.335}},
                          {"H", {0.451, 0.165, -0.083}}};

  // Transform molecular coordinates from angstrom to bohr
  multiply_positions(molecule, angstrom_to_bohr);

  // Get bonds, angles and dihedrals
  std::vector<Bond> B;
  std::vector<Angle> A;
  std::vector<Dihedral> D;
  std::tie(B, A, D) = bad_from_molecule<vec3, vec, mat>(molecule);

  // Check number of bonds
  REQUIRE(B.size() == 5);

  // Check number of angles
  REQUIRE(A.size() == 5);

  // Check number of dihedral angles
  REQUIRE(D.size() == 3);

  // TODO: Check wit other codes (where dihedral 2-1-3-6 is added).

  // Compute IRC
  vec q{
      cartesian_to_irc<vec3, vec>(to_cartesian<vec3, vec>(molecule), B, A, D)};

  // Check number of IRC
  REQUIRE(linalg::size<vec>(q) == 13);

  SECTION("Bonds") {
    // TODO
  }

  SECTION("Angles") {
    // TODO
  }

  SECTION("Dihedrals") {
    // TODO
  }
}

// Quasi-linear angles
TEST_CASE("Connectivity for water dimer") {
  // TODO
}

// Interfragment bonds
TEST_CASE("Connectivity for benzene dimer") {
  // TODO
}

TEST_CASE("Connectivity test for CH2O") {
  // TODO
}

TEST_CASE("Connectivity of molecule database") {
  using namespace io;

  using namespace connectivity;
  using namespace molecule;
  using namespace tools;

  struct ConnectivityResult {
    std::string filename;
    size_t n_bonds;
    size_t n_angles;
    size_t n_dihedrals;
  };

  auto results =
      std::vector<ConnectivityResult>{{"hydrogen_peroxide.xyz", 3, 2, 1},
                                      {"ethanol.xyz", 8, 13, 12},
                                      {"glycerol.xyz", 13, 21, 27},
                                      {"octane.xyz", 25, 48, 63},
                                      {"phenol.xyz", 13, 19, 26},
                                      {"indene.xyz", 18, 30, 44},
                                      {"toluene.xyz", 15, 24, 30},
                                      {"caffeine.xyz", 25, 43, 54}};
  for (const auto &molecule_parameters : results) {
    CAPTURE(molecule_parameters.filename);

    const auto mol =
        load_xyz<vec3>(config::molecules_dir + molecule_parameters.filename);

    std::vector<Bond> B;
    std::vector<Angle> A;
    std::vector<Dihedral> D;
    std::tie(B, A, D) = bad_from_molecule<vec3, vec, mat>(mol);

    CHECK(B.size() == molecule_parameters.n_bonds);
    CHECK(A.size() == molecule_parameters.n_angles);
    CHECK(D.size() == molecule_parameters.n_dihedrals);
  }
}
