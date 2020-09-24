#ifndef IRC_LINALG_H
#define IRC_LINALG_H

#ifdef HAVE_ARMA
#include <armadillo>
#elif HAVE_EIGEN3
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/LU>
#include <unsupported/Eigen/IterativeSolvers>
#else
#error
#endif
#include <iostream>

namespace irc {

namespace linalg {

/// Size of a given container
///
/// \tparam T
/// \param a Container
/// \return Size of \param a
template<typename T>
std::size_t size(const T& a) {
  return a.size();
}

/// Number of rows of a given matrix
///
/// \tparam T
/// \param a Matrix
/// \return Number of rows of \param a
template<typename T>
std::size_t n_rows(const T& a) {
#ifdef HAVE_ARMA
  return a.n_rows;
#elif HAVE_EIGEN3
  return a.rows();
#else
#error
#endif
}

/// Number of columns of a given matrix
///
/// \tparam T
/// \param a Matrix
/// \return Number of columns of \param a
template<typename T>
std::size_t n_cols(const T& a) {
#ifdef HAVE_ARMA
  return a.n_cols;
#elif HAVE_EIGEN3
  return a.cols();
#else
#error
#endif
}

/// Norm of a given vector or matrix
///
/// 2-norm for both vectors and matrices
///
/// \tparam T
/// \return Norm of \param a
template<typename T>
double norm(const T& a) {
#ifdef HAVE_ARMA
  return arma::norm(a, "fro");
#elif HAVE_EIGEN3
  return a.norm();
#else
#error
#endif
}

/// Normalise vector or matrix
///
/// \tparam T
/// \return Normalised version of \param a
template<typename T>
T normalize(const T& a) {
#ifdef HAVE_ARMA
  return arma::normalise(a);
#elif HAVE_EIGEN3
  return a.normalized();
#else
#error
#endif
}

/// Dot product between two vectors
///
/// \tparam T
/// \param a Vector
/// \param b Vector
/// \return Dot product between \param a and \param b
template<typename Vector>
double dot(const Vector& a, const Vector& b) {
#ifdef HAVE_ARMA
  return arma::dot(a, b);
#elif HAVE_EIGEN3
  return a.dot(b);
#else
#error
#endif
}

/// Cross product between two vectors
///
/// \tparam Vector3
/// \param a Vector
/// \param b Vector
/// \return Cross product between \param a and \param b
template<typename Vector3>
Vector3 cross(const Vector3& a, const Vector3& b) {
#ifdef HAVE_ARMA
  return arma::cross(a, b);
#elif HAVE_EIGEN3
  return a.cross(b);
#else
#error
#endif
}

/// Allocate column vector of zeros
///
/// \tparam Vector
/// \param nelements Vector size
/// \return Column vector full of zeros
template<typename Vector>
Vector zeros(std::size_t nelements) {
#ifdef HAVE_ARMA
  return arma::zeros<Vector>(nelements);
#elif HAVE_EIGEN3
  return Vector::Zero(nelements);
#else
#error
#endif
}

/// Allocate matrix of zeros
///
/// \tparam Matrix
/// \param nrows Number of rows
/// \param ncols Number of columns
/// \return Matrix full of zeros
template<typename Matrix>
Matrix zeros(std::size_t nrows, std::size_t ncols) {
#ifdef HAVE_ARMA
  return arma::zeros<Matrix>(nrows, ncols);
#elif HAVE_EIGEN3
  return Matrix::Zero(nrows, ncols);
#else
#error
#endif
}

/// Allocate matrix of ones
/// \tparam Matrix
/// \param nrows Number of rows
/// \param ncols Number of columns
/// \return Matrix full of ones
template<typename Matrix>
Matrix ones(std::size_t nrows, std::size_t ncols) {
#ifdef HAVE_ARMA
  return arma::ones<Matrix>(nrows, ncols);
#elif HAVE_EIGEN3
  return Matrix::Ones(nrows, ncols);
#else
#error
#endif
}

/// Allocate identity matrix
///
/// \tparam Matrix
/// \param n Linear size of the identity matrix
/// \return Identity matrix
template<typename Matrix>
Matrix identity(std::size_t n) {
#ifdef HAVE_ARMA
  return arma::eye(n, n);
#elif HAVE_EIGEN3
  return Matrix::Identity(n, n);
#else
#error
#endif
}

/// Matrix transpose
///
/// \tparam Matrix
/// \param mat Matrix
/// \return Transpose of \param mat
template<typename Matrix>
Matrix transpose(const Matrix& mat) {
#ifdef HAVE_ARMA
  return arma::trans(mat);
#elif HAVE_EIGEN3
  return mat.transpose();
#else
#error
#endif
}

/// Inverse matrix
///
/// \tparam Matrix
/// \param mat Matrix
/// \return Inverse of \param mat
template<typename Matrix>
Matrix inv(const Matrix& mat) {
#ifdef HAVE_ARMA
  return arma::inv(mat);
#elif HAVE_EIGEN3
  return mat.inverse();
#else
#error
#endif
}

/// Pseudo-inverse matrix
///
/// \tparam Matrix
/// \param mat Matrix
/// \return Pseudo-inverse of \param mat
template<typename Matrix>
Matrix pseudo_inverse(const Matrix& mat) {
#ifdef HAVE_ARMA
  return arma::pinv(mat);
#elif HAVE_EIGEN3
  std::cout << "Calculating pseudoinverse" << std::endl;

  Eigen::JacobiSVD<Matrix> svd =
      mat.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
  typename Matrix::Scalar tolerance =
      std::numeric_limits<double>::epsilon() *
      std::max(mat.cols(), mat.rows()) *
      svd.singularValues().array().abs().maxCoeff();
  svd.setThreshold(tolerance);

   //Matrix result = svd.matrixV() * Matrix( (svd.singularValues().array().abs()
   //> /*tolerance*/ 1.0e-10).select(svd.singularValues(). array().inverse(), 0)
   //).asDiagonal() * svd.matrixU().adjoint();
  return svd.solve(Matrix::Identity(mat.rows(), mat.rows()));
  // return cod.pseudoInverse();
#else
#error
#endif
}

class MatrixReplacement;

} // namespace linalg
} // namespace irc

namespace Eigen {
namespace internal {
template<>
struct traits<irc::linalg::MatrixReplacement>
  : public Eigen::internal::traits<Eigen::MatrixXd> {};
} // namespace internal
} // namespace Eigen

namespace irc {
namespace linalg {
class MatrixReplacement : public Eigen::EigenBase<MatrixReplacement> {
public:
  typedef double Scalar;
  typedef double RealScalar;
  typedef int StorageIndex;
  enum {
    ColsAtCompileTime = Eigen::Dynamic,
    MaxColsAtCompileTime = Eigen::Dynamic,
    IsRowMajor = false
  };

  Eigen::Index rows() const { return matTmat_.rows(); }
  Eigen::Index cols() const { return matTmat_.cols(); }

  template<typename Rhs>
  Eigen::Product<MatrixReplacement,Rhs,Eigen::AliasFreeProduct> operator*(const Eigen::MatrixBase<Rhs>& x) const {
    return Eigen::Product<MatrixReplacement,Rhs,Eigen::AliasFreeProduct>(*this, x.derived());
  }

  MatrixReplacement() = default;
  MatrixReplacement(const Eigen::MatrixXd& matrix) : mat_(matrix) {
  matTmat_ = matrix.transpose() * matrix;
  matTmat_ = 0.5 * (matTmat_ + matTmat_.transpose());
  }

  void addRegularizationFactor(double regFactor) { regFactor_ = regFactor; }
  const Eigen::MatrixXd& mTm() const {return matTmat_;}
  double regularization() const {return regFactor_;}

private:
  Eigen::MatrixXd mat_, matTmat_;
  double regFactor_{};
};

} // namespace linalg
} // namespace irc

// Implementation of MatrixReplacement * Eigen::DenseVector though a
// specialization of internal::generic_product_impl:
namespace Eigen {
namespace internal {

template<typename Rhs>
struct generic_product_impl<irc::linalg::MatrixReplacement,
                            Rhs,
                            DenseShape,
                            DenseShape,
                            GemvProduct> // GEMV stands for matrix-vector
  : generic_product_impl_base<irc::linalg::MatrixReplacement,
                              Rhs,
                              generic_product_impl<irc::linalg::MatrixReplacement, Rhs>> {
  typedef typename Product<irc::linalg::MatrixReplacement, Rhs>::Scalar Scalar;

  template<typename Dest>
  static void scaleAndAddTo(Dest& dst,
                            const irc::linalg::MatrixReplacement& lhs,
                            const Rhs& rhs,
                            const Scalar& alpha) {
    // This method should implement "dst += alpha * lhs * rhs" inplace,
    // however, for iterative solvers, alpha is always equal to 1, so let's not
    // bother about it.
    assert(alpha == Scalar(1) && "scaling is not implemented");
    EIGEN_ONLY_USED_FOR_DEBUG(alpha);

    dst.noalias() += lhs.mTm() * rhs;
    dst.noalias() += lhs.regularization() * rhs;
  }
};

} // namespace internal
} // namespace Eigen

namespace irc {
namespace linalg {
/// Class: A Solver object for a linear system.
/// This object allows to solve the problem Ax=b,
/// and does so by computing the pseudo inverse in
/// armadillo or by solving iteratively the problem
/// with the LeastSquaresConjugateGradient method in Eigen.
/// The CompleteOrthogonalDecomposition is a bit faster,
/// but the geometry optimization seems to require more iterations.
/// Comparing the 2 with the same geometry optimizer optimizing a
/// 55 atoms molecule very tightly, COD takes 356 cycles, LSCG 312.
///
/// \tparam Matrix
/// \tparam Vector
/// \param mat Matrix
template<typename Matrix, typename Vector>
class Solver {
#ifdef HAVE_ARMA
public:
  Solver(const Matrix& matrix) : matrix_(arma::pinv(matrix)) {}

  Vector solve(const Vector& rhs) { return invMatrix_ * rhs; }

private:
  const Matrix invMatrix_;
#elif HAVE_EIGEN3
public:
  Solver(const Matrix& matrix)
    : tMatrix_(matrix.transpose()), m_(tMatrix_ * matrix), matRep_(matrix) {
    gmres_ = std::make_unique<Eigen::ConjugateGradient<MatrixReplacement, Eigen::Upper|Eigen::Lower, Eigen::IdentityPreconditioner>>();
    matRep_.addRegularizationFactor(5e-6);
    gmres_->compute(matRep_);
  }
  //: cod_(matrix.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV)) {
  //    typename Matrix::Scalar tolerance =
  //    std::numeric_limits<double>::epsilon() * std::max(matrix.cols(),
  //    matrix.rows()) * cod_.singularValues().array().abs().maxCoeff();
  //    cod_.setThreshold(tolerance);
  //}

  // Vector solve(const Vector& rhs) { return cod_.solve(rhs); }
  Vector solve(const Vector& rhs) {
    Vector result;
    if (guess_)
      result = gmres_->solveWithGuess(tMatrix_ * rhs, *guess_);
    else {
      result = gmres_->solve(tMatrix_ * rhs);
      guess_ = std::make_unique<Vector>(result);
    }
    return result;
  }

private:
  // Eigen::JacobiSVD<Matrix> cod_;
  std::unique_ptr<Eigen::ConjugateGradient<MatrixReplacement, Eigen::Upper | Eigen::Lower, Eigen::IdentityPreconditioner>> gmres_;
  MatrixReplacement matRep_;
  Matrix tMatrix_, m_;
  std::unique_ptr<Vector> guess_;
#else
#error
#endif
};

} // namespace linalg

} // namespace irc

#endif // IRC_LINALG_H_H
