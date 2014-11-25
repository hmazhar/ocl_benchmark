#include <vexcl/devlist.hpp>
#include <vexcl/spmat.hpp>
#include <vexcl/vector.hpp>

#include <blaze/math/CompressedMatrix.h>
#include <blaze/math/DynamicVector.h>
#include <blaze/math/DenseSubvector.h>

#include <blaze/util/serialization/Archive.h>
#include <blaze/math/serialization/MatrixSerializer.h>
#include <blaze/math/serialization/VectorSerializer.h>

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
int main(){
	printf("start execution \n");
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
	blaze::DynamicVector<double> res_cpu(g.size());
	double start_cpu = omp_get_wtime();
	res_cpu = M_invD*gamma;
	res_cpu = D_T*res_cpu;
	double end_cpu = omp_get_wtime();

	
	vex::profiler<> prof;

	vex::vector<double> gamma_gpu(ctx,g.size());
	vex::vector<double> b_gpu(ctx,b.size());
	vex::copy(g,gamma_gpu);
	vex::copy(b,b_gpu);
	vex::vector<double> tmp_gpu(ctx,g.size());
	vex::vector<double> res_gpu(ctx,g.size());
	prof.tic_cpu("GPU");
	double start_gpu = omp_get_wtime();
	tmp_gpu  = _M_invD*gamma_gpu;
	res_gpu  = _D_T*tmp_gpu;
	double end_gpu = omp_get_wtime();
	double tot_time = prof.toc("GPU");
	printf("CPU took %f sec. time.\n", end_cpu-start_cpu);
	printf("GPU took %f sec. time, %f sec time.\n", end_gpu-start_gpu, tot_time);
	printf("Speedup: %f\n", (end_cpu-start_cpu) /(end_gpu-start_gpu));

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