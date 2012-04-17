#include <cstdio> 
#include <iostream>
#include "timing.h"

using namespace std;

template<class IndexType, class ValType> 
struct csr_matrix {
   IndexType nrows;
   IndexType ncols;
   IndexType * rowptr;
   IndexType * colind;
   ValType * data;
  
   void matvec(const ValType * X, ValType * Y);

};


template <class IndexType, class ValType>
void csr_matrix<IndexType, ValType>::matvec(const ValType *X, ValType *Y)
{
    IndexType j,k,indval;
    ValType sum;
    #pragma omp parallel for schedule(dynamic) private (indval,sum,j,k) 
    for(IndexType i = 0; i < nrows; i++){
        indval = rowptr[i];
        sum = 0.0;
	for(j = rowptr[i]; j < rowptr[i+1]; j++){
            k = colind[indval];
	    sum += data[indval] * X[k];
	    indval++;
        }
        Y[i] = sum;
    }
}

template<class IndexType, class ValType> 
struct ell_matrix {
   IndexType nrows;
   IndexType ncols;
   Index nnz;
   IndexType nnz_per_row;
   IndexType * indices;
   ValType * data;
  
   void matvec(const ValType * X, ValType * Y);

};

template<class IndexType, class ValType>
void csr_to_ell(const csr_matrix<IndexType, ValType> csr, 
                ell_matrix<IndexType, ValType> ell,  
		const IndexType num_entries_per_row)
//{
//
    IndexType num_entries = 0;
    for(IndexType i = 0; i < src.num_rows; i++)
        num_entries += std::min<IndexType>(num_entries_per_row, csr.rowptr[i+1] - csr.rowptr[i]); 
//
    ell.nrows = csr.nrows;
    ell.ncols = csr.ncols;
    ell.nnz = num_entries;
    ell.nnz_per_row = num_entries_per_row;
    
    IndexType stride = ell.nrows;
    IndexType taille = ell.nnz * stride;
    ell.indices = new IndexType[ell.nnz_per_row * stride];
    ell.data = new ValueType[ell.nnz_per_row * stride];
    
    std::fill(ell.indices, ell.indices + taille, IndexType(0));
    std::fill(ell.data, ell.data + taille,  ValueType(0));

    for(IndexType i = 0; i < csr.num_rows; i++)
    {
        IndexType n = 0;
        IndexType jj = csr.rowptr[i];

        // copy up to num_cols_per_row values of row i into the ELL
        while(jj < src.rowptr[i+1] && n < num_entries_per_row)
        {
            ell.indices[n * stride +i]  = csr.colind[jj];
            ell.values[n * stride +i]  = csr.data[jj];
            jj++, n++;
        }
    }
}



int main(int argc, char * argv[])
{
    
    csr_matrix<unsigned long,double> mat_c;

    cin >> mat_c.nrows >> mat_c.ncols;

    cout << mat_c.nrows  << " "<< mat_c.ncols << std::endl;

    mat_c.rowptr = new unsigned long [mat_c.nrows +1];
    fread(mat_c.rowptr, sizeof(unsigned long), mat_c.nrows + 1, stdin);
    
    unsigned long nnz = mat_c.rowptr[mat_c.nrows];
    cout  << "nnz= "<< nnz << std::endl;
    
    mat_c.colind = new unsigned long [nnz];
    fread(mat_c.colind,sizeof(unsigned long), nnz, stdin);
    
    mat_c.data = new double [nnz];
    fread(mat_c.data,sizeof(double),nnz,stdin);
    
    double * x =  new double [mat_c.ncols];
    double * y =  new double [mat_c.nrows];
    
    // initialization de x
    
    for (unsigned long i = 0 ; i < mat_c.nrows ; ++i) {
              x[i]= 1.0;
    }
    
    tick_t t1,t2;
    timing_init();
    GET_TICK(t1);
    unsigned long nb_tries=300;
    
    for(unsigned long i = 0; i < nb_tries ; ++i) {
       mat_c.matvec(x,  y);
    }
  
    GET_TICK(t2);
    double e1 =  TIMING_DELAY(t1, t2); 
    printf("tps pour %ld runs = %.f s\n",nb_tries, e1/1.e6);
    
    delete [] y;
    delete [] x;
    delete [] mat_c.data;
    delete [] mat_c.colind;
    delete [] mat_c.rowptr;

    return 0;
}

