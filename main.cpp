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

void ConvertSpMat(std::vector<Eigen::Triplet<double> > & triplet, blaze::CompressedMatrix<double>& mat) {
	uint non_zeros = mat.nonZeros();
	triplet.clear();
	triplet.reserve(non_zeros);

	for (int i = 0; i < mat.rows(); ++i) {
		for (blaze::CompressedMatrix<double>::Iterator it = mat.begin(i); it != mat.end(i); ++it) {
			triplet.push_back(Eigen::Triplet<double>(i,it->index(),it->value()));
		}
	}
}


std::string slurp(std::string fname){
	std::ifstream file(fname.c_str());
	std::string buffer((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
	return buffer;
}


void readSPMAT(std::string & data, blaze::CompressedMatrix<double> & mat){
	std::stringstream ss(data);

	uint rows, cols, nonzeros;
	ss>>rows>>cols>>nonzeros;
	//printf("INFO: rows: %d, columns: %d, non zeros: %d\n",rows, cols, nonzeros);
	mat.resize(rows,cols);
	mat.reserve(nonzeros);
	uint row_nonzeros;
	uint colval;
	double v;

	for(int i=0; i<rows; i++){
		if(i%1000000==0){printf("%d, ",i);}
		
		ss>>row_nonzeros;

		for(int j=0; j<row_nonzeros; j++){
			ss>>colval>>v;
			mat.append(i,colval,v);
		}
		mat.finalize(i);
	}
	printf("\n");
}
void readVector(std::string & data, std::vector<double> & vec){
	std::stringstream ss(data);

	uint size;
	ss>>size;
	vec.resize(size);

	//printf("INFO: size: %d\n",size);
	double v;
	for(int i=0; i<size; i++){
		ss>>v;
		vec[i] = v;
	}
}
int main(int argc, char* argv[]){
	vex::profiler<> prof;
	vex::Context ctx(vex::Filter::Env && vex::Filter::DoublePrecision);
	std::cout<<ctx<<std::endl;
	printf("create context \n");


	blaze::CompressedMatrix<double> D_T, M_invD;
	std::vector<double> g,b;
	printf("read D^T \n");
	{
		std::string data = slurp("D_T.mat");
		readSPMAT(data, D_T);
	}

	{
		std::string data = slurp("M_invD.mat");
		readSPMAT(data, M_invD);
	}
	{
		std::string data = slurp("gamma.vec");
		readVector(data, g);
	}

	{
		std::string data = slurp("b.vec");
		readVector(data, b);
	}


	printf("D_T: rows: %d, columns: %d, non zeros: %d\n", D_T.rows(), D_T.columns(), D_T.nonZeros());
	printf("M_invD: rows: %d, columns: %d, non zeros: %d\n", M_invD.rows(), M_invD.columns(), M_invD.nonZeros());
	printf("gamma: size: %d\n", g.size());
	printf("b: size: %d\n", b.size());

	std::vector<size_t> row;
	std::vector<size_t> col;
	std::vector<double> val;

	ConvertSpMat(row, col, val, D_T);
	vex::SpMat<double> _D_T(ctx, D_T.rows(), D_T.columns(), row.data(), col.data(), val.data());

	ConvertSpMat(row, col, val, M_invD);
	vex::SpMat<double> _M_invD(ctx, M_invD.rows(), M_invD.columns(), row.data(), col.data(), val.data());

	blaze::DynamicVector<double> gamma(g.size()),rhs(g.size());

	for(int i=0; i<g.size(); i++){
		gamma[i] = g[i];
		rhs[i] = b[i];
	}
	blaze::DynamicVector<double> tmp_cpu(g.size());
	blaze::DynamicVector<double> res_cpu(g.size());

	prof.tic_cpu("CPU");


	double RUNS = 1;
	if(argc>1){
		RUNS=atoi(argv[1]);
	}
	for(size_t i = 0; i < RUNS; i++){
		tmp_cpu = M_invD*gamma;
		res_cpu += D_T*tmp_cpu;
	}
double cpu_time = prof.toc("CPU");
	
	

	vex::vector<double> gamma_gpu(ctx,g.size());
	vex::vector<double> b_gpu(ctx,b.size());
	vex::copy(g,gamma_gpu);
	vex::copy(b,b_gpu);
	vex::vector<double> tmp_gpu(ctx,g.size());
	vex::vector<double> res_gpu(ctx,g.size());


	tmp_gpu += _M_invD*gamma_gpu;
	tmp_gpu = 0;
	res_gpu += _D_T*tmp_gpu;
	res_gpu = 0;
	prof.tic_cpu("GPU");
	double start_gpu = omp_get_wtime();
	for(size_t i = 0; i < RUNS; i++){
		tmp_gpu  = _M_invD*gamma_gpu;
		res_gpu  += _D_T*tmp_gpu;
	}
	ctx.finish();
	double end_gpu = omp_get_wtime();
	double gpu_time = prof.toc("GPU");
	//Two Spmv each with 2*nnz operations
	uint operations = 2 * 2 * D_T.nonZeros();
	uint moved = 2* (D_T.nonZeros() + 2* D_T.rows()) * sizeof(double);





	std::vector<Eigen::Triplet<double> >  triplet;

	ConvertSpMat(triplet, D_T);
	Eigen::SparseMatrix<double> eigen_D_T(D_T.rows(),D_T.columns());
	eigen_D_T.setFromTriplets(triplet.begin(), triplet.end());


ConvertSpMat(triplet, M_invD);
	Eigen::SparseMatrix<double> eigen_M_invD(M_invD.rows(),M_invD.columns());
	eigen_M_invD.setFromTriplets(triplet.begin(), triplet.end());

Eigen::VectorXd eigen_gamma(g.size());
Eigen::VectorXd eigen_rhs(g.size());

Eigen::VectorXd tmp_eigen(g.size());
Eigen::VectorXd res_eigen(g.size());

for(int i=0; i<g.size(); i++){
		eigen_gamma[i] = g[i];
		eigen_rhs[i] = b[i];
	}

prof.tic_cpu("EIGEN"); 
for(size_t i = 0; i < RUNS; i++){
	tmp_eigen = eigen_M_invD*eigen_gamma;
	res_eigen += eigen_D_T*tmp_eigen;
}
double eigen_time = prof.toc("EIGEN");



	double cpu_single = cpu_time/RUNS;
	double gpu_single = gpu_time/RUNS;
	double eig_single = eigen_time/RUNS;
	double GFLOP = 1000000000;
	printf("Blaze %f sec. Eigen %f sec. OCL %f sec.\n", cpu_single, eig_single, gpu_single);
	printf("Speedup: Blaze vs Eigen %f\n", cpu_single /eig_single);
	printf("Speedup: Blaze vs OCL %f\n", cpu_single /gpu_single);
	printf("Speedup: Eigen vs OCL %f\n", eig_single /gpu_single);

	printf("Flops: Blaze: %f Eigen %f OCL: %f\n", operations/cpu_single/GFLOP,operations/eig_single/GFLOP,operations/gpu_single/GFLOP);
	printf("Bandwidth: Blaze: %f Eigen %f OCL: %f\n", moved/cpu_single/GFLOP,moved/eig_single/GFLOP, moved /gpu_single/GFLOP);


	//std::vector<double> res_host(g.size());

	//vex::copy(res_gpu,res_host);





	//printf("copy\n");
	// for(int i=0; i<g.size(); i++){
	// 	if(res_host[i]!=res_cpu[i]){
	// 		printf("%f\n",res_host[i]-res_cpu[i]);
	// 	}
	// }



	return 0;
}