#include <vexcl/devlist.hpp>
#include <vexcl/spmat.hpp>
#include <vexcl/vector.hpp>

#include <blaze/math/CompressedMatrix.h>
#include <blaze/math/DynamicVector.h>
#include <blaze/math/DenseSubvector.h>

#include <blaze/util/serialization/Archive.h>
#include <blaze/math/serialization/MatrixSerializer.h>
#include <blaze/math/serialization/VectorSerializer.h>

#include <Eigen/Sparse>

#include <vector>
#include <sstream>
#include <string>
#include <iostream>

void ConvertSpMat(std::vector<size_t>& row, std::vector<size_t>& col, std::vector<double>& val, blaze::CompressedMatrix<double>& mat) {
  uint non_zeros = mat.nonZeros();
  row.clear();
  col.clear();
  val.clear();
  uint count = 0;
  for (int i = 0; i < mat.rows(); ++i) {
    row.push_back(count);
    for (blaze::CompressedMatrix<double>::Iterator it = mat.begin(i); it != mat.end(i); ++it) {
      col.push_back(it->index());
      val.push_back(it->value());
      count++;
    }
  }
  row.push_back(count);
}

void ConvertSpMat(std::vector<Eigen::Triplet<double> >& triplet, blaze::CompressedMatrix<double>& mat) {
  uint non_zeros = mat.nonZeros();
  triplet.clear();
  triplet.reserve(non_zeros);

  for (int i = 0; i < mat.rows(); ++i) {
    for (blaze::CompressedMatrix<double>::Iterator it = mat.begin(i); it != mat.end(i); ++it) {
      triplet.push_back(Eigen::Triplet<double>(i, it->index(), it->value()));
    }
  }
}

std::string slurp(std::string fname) {
  std::ifstream file(fname.c_str());
  std::string buffer((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  return buffer;
}

void readSPMAT(const std::string filename, blaze::CompressedMatrix<double>& mat) {
  std::stringstream ss(slurp(filename));

  uint rows, cols, nonzeros;
  uint row_nonzeros, colval;
  ss >> rows >> cols >> nonzeros;
  mat.resize(rows, cols);
  mat.reserve(nonzeros);

  double v;
  for (int i = 0; i < rows; i++) {
    ss >> row_nonzeros;
    for (int j = 0; j < row_nonzeros; j++) {
      ss >> colval >> v;
      mat.append(i, colval, v);
    }
    mat.finalize(i);
  }
}
void readVector(const std::string filename, blaze::DynamicVector<double>& vec) {
  std::stringstream ss(slurp(filename));

  uint size;
  ss >> size;
  vec.resize(size);

  double v;
  for (int i = 0; i < size; i++) {
    ss >> v;
    vec[i] = v;
  }
}

blaze::CompressedMatrix<double> D_T_blaze, M_invD_blaze;
blaze::DynamicVector<double> gamma_blaze, rhs_blaze;

int num_rows;
int num_cols;
int num_nonzeros;
int RUNS = 1;

vex::profiler<> prof;

double blaze_time;
double eigen_time;
double vexcl_time;

template <typename Matrix, typename Vector>
void TEST(Matrix& D_T, Matrix& M_invD, Vector& gamma, Vector& temporary, Vector& result) {
  for (size_t i = 0; i < RUNS; i++) {
    temporary = M_invD * gamma;
    result += D_T * temporary;
  }
}

void Blaze_TEST() {
  blaze::DynamicVector<double> temporary(num_rows);
  blaze::DynamicVector<double> result(num_rows);

  prof.tic_cpu("Blaze");
  TEST(D_T_blaze, M_invD_blaze, gamma_blaze, temporary, result);
  blaze_time = prof.toc("Blaze");
}

void VexCL_TEST() {

  vex::Context ctx(vex::Filter::Env && vex::Filter::DoublePrecision);

  std::vector<size_t> row, col;
  std::vector<double> val;

  ConvertSpMat(row, col, val, D_T_blaze);
  vex::SpMat<double> D_T_vex(ctx, num_rows, num_cols, row.data(), col.data(), val.data());

  ConvertSpMat(row, col, val, M_invD_blaze);
  vex::SpMat<double> M_invD_vex(ctx, num_rows, num_cols, row.data(), col.data(), val.data());

  std::vector<double> gamma_temp(num_rows);

  for (int i = 0; i < num_rows; i++) {
    gamma_temp[i] = gamma_blaze[i];
  }

  vex::vector<double> gamma_vex(ctx, num_rows);
  vex::copy(gamma_temp, gamma_vex);
  vex::vector<double> temporary(ctx, num_rows);
  vex::vector<double> result(ctx, num_rows);

  temporary += M_invD_vex * gamma_vex;
  temporary = 0;
  result += D_T_vex * temporary;
  result = 0;

  prof.tic_cpu("VexCL");
  TEST(D_T_vex, M_invD_vex, gamma_vex, temporary, result);
  ctx.finish();
  vexcl_time = prof.toc("VexCL");
}

void Eigen_TEST() {

  std::vector<Eigen::Triplet<double> > triplet;

  ConvertSpMat(triplet, D_T_blaze);
  Eigen::SparseMatrix<double> D_T_eigen(num_rows, num_cols);
  D_T_eigen.setFromTriplets(triplet.begin(), triplet.end());

  ConvertSpMat(triplet, M_invD_blaze);
  Eigen::SparseMatrix<double> M_invD_eigen(num_rows, num_cols);
  M_invD_eigen.setFromTriplets(triplet.begin(), triplet.end());

  Eigen::VectorXd gamma_eigen(num_rows);
  Eigen::VectorXd temporary(num_rows);
  Eigen::VectorXd result(num_rows);

  for (int i = 0; i < num_rows; i++) {
    gamma_eigen[i] = gamma_blaze[i];
  }
  prof.tic_cpu("EIGEN");
  TEST(D_T_eigen, M_invD_eigen, gamma_eigen, temporary, result);
  eigen_time = prof.toc("EIGEN");
}

int main(int argc, char* argv[]) {
  if (argc > 2) {
    RUNS = atoi(argv[2]);
  }

  readSPMAT("D_T_"+std::string(argv[1])+".dat", D_T_blaze);
  readSPMAT("M_invD_"+std::string(argv[1])+".dat", M_invD_blaze);
  readVector("gamma_"+std::string(argv[1])+".dat", gamma_blaze);
  readVector("b_"+std::string(argv[1])+".dat", rhs_blaze);

  num_rows = D_T_blaze.rows();
  num_cols = D_T_blaze.columns();
  num_nonzeros = D_T_blaze.nonZeros();

  printf("D_T: rows: %d, columns: %d, non zeros: %d\n", D_T_blaze.rows(), D_T_blaze.columns(), D_T_blaze.nonZeros());
  printf("M_invD: rows: %d, columns: %d, non zeros: %d\n", M_invD_blaze.rows(), M_invD_blaze.columns(), M_invD_blaze.nonZeros());
  printf("gamma: size: %d\n", gamma_blaze.size());
  printf("b: size: %d\n", rhs_blaze.size());

  Blaze_TEST();
  Eigen_TEST();
  VexCL_TEST();

  // Two Spmv each with 2*nnz operations
  uint operations = 2 * 2 * num_nonzeros;
  uint moved = 2 * (num_nonzeros + 2 * num_rows) * sizeof(double);

  double blaze_single = blaze_time / RUNS;
  double vex_single = vexcl_time / RUNS;
  double eig_single = eigen_time / RUNS;
  double GFLOP = 1000000000;
  printf("Blaze %f sec. Eigen %f sec. OCL %f sec.\n", blaze_single, eig_single, vex_single);
  printf("Speedup: Blaze vs Eigen %f\n", blaze_single / eig_single);
  printf("Speedup: Blaze vs OCL %f\n", blaze_single / vex_single);
  printf("Speedup: Eigen vs OCL %f\n", eig_single / vex_single);

  printf("Flops: Blaze: %f Eigen %f VexCL: %f\n", operations / blaze_single / GFLOP, operations / eig_single / GFLOP, operations / vex_single / GFLOP);
  printf("Bandwidth: Blaze: %f Eigen %f VexCL: %f\n", moved / blaze_single / GFLOP, moved / eig_single / GFLOP, moved / vex_single / GFLOP);

  return 0;
}
