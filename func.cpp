#include <stdio.h>
#include <stdlib.h>
#include <mkl.h>
#include <mpi.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include <bitset>
#include <map>
#include "func.h"

using std::vector;
using std::pair;
using std::make_pair;
using std::bitset;
using std::map;
using std::max;



extern int output(int rows, int *row_ptr, int *col_idx, double *val);
extern void Qsort(int *a, double *b, int low, int high);


void transpose(CSR *A, CSR *AT) {
    AT->rows = A->cols;
    AT->cols = A->rows;
    AT->nnz = A->nnz;
    AT->row_ptr = (int*)malloc((AT->rows+1)*sizeof(int));
    AT->col_idx = (int*)malloc(AT->nnz*sizeof(int));
    AT->value = (double*)malloc(AT->nnz*sizeof(double));

    vector<pair<int, double> > *t = new vector<pair<int, double> >[A->rows];

    int cnt = 0;
    for (int i = 0; i < A->rows; ++i) {
        for (int j = A->row_ptr[i]; j < A->row_ptr[i+1]; ++j) {
            t[A->col_idx[cnt]].push_back(make_pair(i, A->value[cnt]));
            ++cnt;
        }
    }

    cnt = 0;
    for (int i = 0; i < A->rows; ++i) {
        AT->row_ptr[i] = cnt;
        vector<pair<int, double> >::iterator it = t[i].begin();
        for (; it != t[i].end(); ++it) {
            AT->col_idx[cnt] = it->first;
            AT->value[cnt] = it->second;
            ++cnt;
        }
    }
    AT->row_ptr[A->rows] = cnt;
}

void add(CSR *A, CSR *B, CSR *C) {

    vector<pair<int, double> > *t = new vector<pair<int, double> >[A->rows];

    int nnz = 0;
    for (int x = 0; x < A->rows; x++) {
        double value;
        int y;
        int pta = A->row_ptr[x];
        int ptb = B->row_ptr[x];
        while (pta < A->row_ptr[x+1] && ptb < B->row_ptr[x+1]) {
            int ya = A->col_idx[pta];
            int yb = B->col_idx[ptb];
            if (ya == yb) {
                value = A->value[pta]+B->value[ptb];
                y = ya;
                ++pta;
                ++ptb;
            } else if (ya < yb) {
                value = A->value[pta];
                y = ya;
                ++pta;
            } else {
                value = A->value[pta];
                y = yb;
                ++ptb;
            }
            if (value != 0) {
                t[x].push_back(make_pair(y, value));
                ++nnz;
            }
        }
        while (pta < A->row_ptr[x+1]) {
            int ya = A->col_idx[pta];
            value = A->value[pta];
            t[x].push_back(make_pair(ya, value));
            ++nnz;
            ++pta;
        }
        while (ptb < B->row_ptr[x+1]) {
            int yb = B->col_idx[ptb];
            value = B->value[ptb];
            t[x].push_back(make_pair(yb, value));
            ++nnz;
            ++ptb;
        }
    }

    C->rows = A->rows;
    C->cols = A->cols;
    C->nnz = nnz;
    C->row_ptr = (int*)malloc((C->rows+1)*sizeof(int));
    C->col_idx = (int*)malloc((C->nnz)*sizeof(int));
    C->value = (double*)malloc((C->nnz)*sizeof(double));

    int cnt = 0;
    for (int i = 0; i < C->rows; ++i) {
        C->row_ptr[i] = cnt;
        vector<pair<int, double> >::iterator it = t[i].begin();
        for (; it != t[i].end(); ++it) {
            C->col_idx[cnt] = it->first;
            C->value[cnt] = it->second;
            ++cnt;
        }
    }
    C->row_ptr[C->rows] = cnt;
}


struct node {
    int col;
    double value;
};

node *nodes;
node **new_nodes;

void multi(CSR *A, CSR *B, CSR *C, bool first) {

    vector<pair<int, double> > *t = new vector<pair<int, double> >[A->rows];

    int *nnzs = (int*)malloc(A->rows*sizeof(int));
    memset(nnzs, 0, A->rows*sizeof(int));


    if (first && false) {
        nodes = (node *)malloc(A->nnz*sizeof(node));
        for (int i = 0; i < A->nnz; ++i) {
            nodes[i].col = A->col_idx[i];
            nodes[i].value = A->value[i];
        }

        /*
        CSR AT;
        transpose(A, &AT);
        int max_num = 0;
        for (int x = 0; x < AT->rows; ++x) {
            max_num = max(max_num, AT->row_ptr[x+1]-A->row_ptr[x]);
        }*/
    }


    /*
#pragma omp parallel
    {
#pragma omp single
        {

            for (int x1 = 0; x1 < A->rows; ++x1) {
                if (A->row_ptr[x1] == A->row_ptr[x1+1]) {
                    continue;
                }
#pragma omp task
                for (int x2 = 0; x2 < B->rows; ++x2) {
                    double sum = 0;
                    map<int, double>::iterator it;
                    for (int idx = B->row_ptr[x2]; idx < B->row_ptr[x2+1]; ++idx) {
                        if ((it = m[x1].find(B->col_idx[idx])) != m[x1].end()) {
                            sum += it->second*B->value[idx];
                        }
                    }
                    if (sum != 0) {
                        t[x1].push_back(make_pair(x2, sum));
                        ++nnzs[x1];
                    }
                }
            }
#pragma omp taskwait
        }
    }*/

/*#pragma omp parallel
    {
#pragma omp single
        {
            for (int x1 = 0; x1 < A->rows; ++x1) {
                if (A->row_ptr[x1] == A->row_ptr[x1+1]) {
                    continue;
                }
#pragma omp task
                for (int x2 = 0; x2 < B->rows; ++x2) {
                    double sum = 0;
                    int pta = A->row_ptr[x1];
                    int ptb = B->row_ptr[x2];
                    int ya;
                    int yb;
                    while (pta < A->row_ptr[x1+1] && ptb < B->row_ptr[x2+1]) {
                        ya = nodes[pta].col;
                        yb = nodes[ptb].col;
                        if (ya == yb) {
                            sum += nodes[pta].value*nodes[ptb].value;
                            ++pta;
                            ++ptb;
                        } else if (ya < yb) {
                            ++pta;
                        } else {
                            ++ptb;
                        }
                    }
                    if (sum != 0) {
                        t[x1].push_back(make_pair(x2, sum));
                        ++nnzs[x1];
                    }
                }
            }
#pragma omp taskwait
        }
    }*/

#pragma omp parallel
    {
#pragma omp single
        {
            for (int x1 = 0; x1 < A->rows; ++x1) {
                if (A->row_ptr[x1] == A->row_ptr[x1+1]) {
                    continue;
                }
#pragma omp task
                for (int x2 = 0; x2 < B->rows; ++x2) {
                    double sum = 0;
                    int pta = A->row_ptr[x1];
                    int ptb = B->row_ptr[x2];
                    int ya;
                    int yb;
                    while (pta < A->row_ptr[x1+1] && ptb < B->row_ptr[x2+1]) {
                        ya = A->col_idx[pta];
                        yb = B->col_idx[ptb];
                        if (ya == yb) {
                            sum += A->value[pta]*B->value[ptb];
                            ++pta;
                            ++ptb;
                        } else if (ya < yb) {
                            ++pta;
                        } else {
                            ++ptb;
                        }
                    }
                    if (sum != 0) {
                        t[x1].push_back(make_pair(x2, sum));
                        ++nnzs[x1];
                    }
                }
            }
#pragma omp taskwait
        }
    }

    int nnz = 0;
    for (int i = 0; i < A->rows; ++i) {
        nnz += nnzs[i];
    }

    C->rows = A->rows;
    C->cols = A->cols;
    C->nnz = nnz;
    C->row_ptr = (int*)malloc((C->rows+1)*sizeof(int));
    C->col_idx = (int*)malloc((C->nnz)*sizeof(int));
    C->value = (double*)malloc((C->nnz)*sizeof(double));

    int cnt = 0;
    for (int i = 0; i < C->rows; ++i) {
        C->row_ptr[i] = cnt;
        vector<pair<int, double> >::iterator it = t[i].begin();
        for (; it != t[i].end(); ++it) {
            C->col_idx[cnt] = it->first;
            C->value[cnt] = it->second;
            ++cnt;
        }
    }
    C->row_ptr[C->rows] = cnt;
}

/* B = AT+A */
int sparse_AT_plus_A_type0(CSR *B, CSR *A, bool first_time)
{
    if(first_time) /* The first time to use Matrix A, you can do some pre-process. */
    {
    }
    int rows =  A->rows;
    int cols =  A->cols;
    int nnz  =  A->nnz;
    int rank = A->rank;
    int nprocs = A->nprocs;

    if(rank == 0)
    {
        int *rows_per_proc = (int*)malloc(nprocs*sizeof(int));
        int *row_ptr  = (int*)malloc((rows+1)*sizeof(int));
        int *pointerE = NULL;//(int*)malloc(rows*sizeof(int));
        int *col_idx  = (int*)malloc(nnz*sizeof(int));
        double *value = (double*)malloc(nnz*sizeof(double));
        memcpy(row_ptr, A->row_ptr, (A->rows_local+1)*sizeof(int));
        memcpy(col_idx, A->col_idx, A->nnz_local*sizeof(int));
        memcpy(value,   A->value,   A->nnz_local*sizeof(double));
        int rows_offset = A->rows_local;
        int nnz_offset = A->nnz_local;
        rows_per_proc[0] = A->rows_local;
        for(int i = 1; i < nprocs; i ++)
        {
            int rows_local;
            int nnz_local;
            MPI_Status status;
            MPI_Recv(&rows_local, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
            MPI_Recv(&nnz_local,  1, MPI_INT, i, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(row_ptr + rows_offset, rows_local+1, MPI_INT, i, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(col_idx + nnz_offset,  nnz_local,    MPI_INT, i, 3, MPI_COMM_WORLD, &status);
            MPI_Recv(value   + nnz_offset,  nnz_local, MPI_DOUBLE, i, 4, MPI_COMM_WORLD, &status);       
            rows_offset += rows_local;
            nnz_offset += nnz_local;
            rows_per_proc[i] = rows_local;
        }

        CSR a;
        a.rows = rows;
        a.cols = cols;
        a.nnz = nnz;
        a.row_ptr = row_ptr;
        a.col_idx = col_idx;
        a.value = value;
        CSR at;
        transpose(&a, &at);
        CSR c;
        add(&at, &a, &c);

        free(row_ptr); row_ptr = NULL;
        free(col_idx); col_idx = NULL;
        free(value);   value   = NULL;

        rows = c.rows;
        cols = c.cols;
        row_ptr = c.row_ptr;
        col_idx = c.col_idx;
        value = c.value;


        if(first_time)
        {
            B->rank = rank;
            B->nprocs = nprocs;
            B->rows = rows;
            B->cols = cols;
            B->nnz = row_ptr[rows] - row_ptr[0];
            B->rows_local  = rows_per_proc[0];
            B->rows_offset = 0;
            B->nnz_local   = row_ptr[B->rows_local] - row_ptr[0];
            B->row_ptr = (int*)malloc((B->rows_local+1)*sizeof(int));
            B->col_idx = (int*)malloc(B->nnz_local*sizeof(int));
            B->value = (double*)malloc(B->nnz_local*sizeof(double));
            rows_offset = B->rows_local;
            int total_nnz = row_ptr[rows] - row_ptr[0];
            for(int i = 1; i < nprocs; i ++)
            {
                int rows_local = rows_per_proc[i];
                int nnz_local = row_ptr[rows_offset+rows_local] - row_ptr[rows_offset];
                MPI_Send(&total_nnz, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                MPI_Send(&rows_local, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
                MPI_Send(&rows_offset, 1, MPI_INT, i, 2, MPI_COMM_WORLD);
                MPI_Send(&nnz_local, 1, MPI_INT, i, 3, MPI_COMM_WORLD);
                rows_offset += rows_local;
            }
        }
        memcpy(B->row_ptr, row_ptr, (rows_per_proc[0]+1)*sizeof(int));
        memcpy(B->col_idx, col_idx, (row_ptr[B->rows_local]-row_ptr[0])*sizeof(int));
        memcpy(B->value,   value,   (row_ptr[B->rows_local]-row_ptr[0])*sizeof(double));
        rows_offset = B->rows_local;
        for(int i = 1; i < nprocs; i ++)
        {
            int rows_local = rows_per_proc[i];
            MPI_Send(row_ptr + rows_offset, rows_local+1, MPI_INT, i, 4, MPI_COMM_WORLD);
            MPI_Send(col_idx + row_ptr[rows_offset], row_ptr[rows_offset+rows_local]-row_ptr[rows_offset], MPI_INT,    i, 5, MPI_COMM_WORLD);
            MPI_Send(value   + row_ptr[rows_offset], row_ptr[rows_offset+rows_local]-row_ptr[rows_offset], MPI_DOUBLE, i, 6, MPI_COMM_WORLD);
            rows_offset += rows_local;
        }

        free(rows_per_proc);
    }
    else
    {
        MPI_Status status;
        MPI_Send(&A->rows_local, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&A->nnz_local,  1, MPI_INT, 0, 1, MPI_COMM_WORLD);
        MPI_Send(A->row_ptr, A->rows_local+1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(A->col_idx, A->nnz_local,    MPI_INT, 0, 3, MPI_COMM_WORLD);
        MPI_Send(A->value,   A->nnz_local, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD);
        if(first_time)
        {
            B->rank = rank;
            B->nprocs = nprocs;
            B->rows = rows;
            B->cols = cols;
            MPI_Recv(&B->nnz,         1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
            MPI_Recv(&B->rows_local,  1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(&B->rows_offset, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(&B->nnz_local,   1, MPI_INT, 0, 3, MPI_COMM_WORLD, &status);
            B->row_ptr = (int*)malloc((B->rows_local+1)*sizeof(int));
            B->col_idx = (int*)malloc(B->nnz_local*sizeof(int));
            B->value = (double*)malloc(B->nnz_local*sizeof(double));
        }
        MPI_Recv(B->row_ptr, B->rows_local+1, MPI_INT,    0, 4, MPI_COMM_WORLD, &status);
        MPI_Recv(B->col_idx, B->nnz_local,    MPI_INT,    0, 5, MPI_COMM_WORLD, &status);
        MPI_Recv(B->value,   B->nnz_local,    MPI_DOUBLE, 0, 6, MPI_COMM_WORLD, &status);
    }
    return 0;
}

/* B = AT+A */
int sparse_AT_plus_A_type1(CSR *B, CSR *A, bool first_time)
{
    if(first_time) /* The first time to use Matrix A, you can do some pre-process. */
    {

    }
    int rows =  A->rows;
    int cols =  A->cols;
    int nnz  =  A->nnz;
    int rank = A->rank;
    int nprocs = A->nprocs;

    if(rank == 0)
    {
        int *rows_per_proc = (int*)malloc(nprocs*sizeof(int));
        int *row_ptr  = (int*)malloc((rows+1)*sizeof(int));
        int *pointerE = NULL;//(int*)malloc(rows*sizeof(int));
        int *col_idx  = (int*)malloc(nnz*sizeof(int));
        double *value = (double*)malloc(nnz*sizeof(double));
        memcpy(row_ptr, A->row_ptr, (A->rows_local+1)*sizeof(int));
        memcpy(col_idx, A->col_idx, A->nnz_local*sizeof(int));
        memcpy(value,   A->value,   A->nnz_local*sizeof(double));
        int rows_offset = A->rows_local;
        int nnz_offset = A->nnz_local;
        rows_per_proc[0] = A->rows_local;
        for(int i = 1; i < nprocs; i ++)
        {
            int rows_local;
            int nnz_local;
            MPI_Status status;
            MPI_Recv(&rows_local, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
            MPI_Recv(&nnz_local,  1, MPI_INT, i, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(row_ptr + rows_offset, rows_local+1, MPI_INT, i, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(col_idx + nnz_offset,  nnz_local,    MPI_INT, i, 3, MPI_COMM_WORLD, &status);
            MPI_Recv(value   + nnz_offset,  nnz_local, MPI_DOUBLE, i, 4, MPI_COMM_WORLD, &status);       
            rows_offset += rows_local;
            nnz_offset += nnz_local;
            rows_per_proc[i] = rows_local;
        }
        CSR a;
        a.rows = rows;
        a.cols = cols;
        a.nnz = nnz;
        a.row_ptr = row_ptr;
        a.col_idx = col_idx;
        a.value = value;
        CSR at;
        transpose(&a, &at);
        CSR c;
        add(&at, &a, &c);

        free(row_ptr); row_ptr = NULL;
        free(col_idx); col_idx = NULL;
        free(value);   value   = NULL;

        rows = c.rows;
        cols = c.cols;
        row_ptr = c.row_ptr;
        col_idx = c.col_idx;
        value = c.value;


        if(first_time)
        {
            B->rank = rank;
            B->nprocs = nprocs;
            B->rows = rows;
            B->cols = cols;
            B->nnz = row_ptr[rows] - row_ptr[0];
            B->rows_local  = rows_per_proc[0];
            B->rows_offset = 0;
            B->nnz_local   = row_ptr[B->rows_local] - row_ptr[0];
            B->row_ptr = (int*)malloc((B->rows_local+1)*sizeof(int));
            B->col_idx = (int*)malloc(B->nnz_local*sizeof(int));
            B->value = (double*)malloc(B->nnz_local*sizeof(double));
            rows_offset = B->rows_local;
            int total_nnz = row_ptr[rows] - row_ptr[0];
            for(int i = 1; i < nprocs; i ++)
            {
                int rows_local = rows_per_proc[i];
                int nnz_local = row_ptr[rows_offset+rows_local] - row_ptr[rows_offset];
                MPI_Send(&total_nnz, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                MPI_Send(&rows_local, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
                MPI_Send(&rows_offset, 1, MPI_INT, i, 2, MPI_COMM_WORLD);
                MPI_Send(&nnz_local, 1, MPI_INT, i, 3, MPI_COMM_WORLD);
                rows_offset += rows_local;
            }
        }
        memcpy(B->row_ptr, row_ptr, (rows_per_proc[0]+1)*sizeof(int));
        memcpy(B->col_idx, col_idx, (row_ptr[B->rows_local]-row_ptr[0])*sizeof(int));
        memcpy(B->value,   value,   (row_ptr[B->rows_local]-row_ptr[0])*sizeof(double));
        rows_offset = B->rows_local;
        for(int i = 1; i < nprocs; i ++)
        {
            int rows_local = rows_per_proc[i];
            MPI_Send(row_ptr + rows_offset, rows_local+1, MPI_INT, i, 4, MPI_COMM_WORLD);
            MPI_Send(col_idx + row_ptr[rows_offset], row_ptr[rows_offset+rows_local]-row_ptr[rows_offset], MPI_INT,    i, 5, MPI_COMM_WORLD);
            MPI_Send(value   + row_ptr[rows_offset], row_ptr[rows_offset+rows_local]-row_ptr[rows_offset], MPI_DOUBLE, i, 6, MPI_COMM_WORLD);
            rows_offset += rows_local;
        }

        free(rows_per_proc);
    }
    else
    {
        MPI_Status status;
        MPI_Send(&A->rows_local, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&A->nnz_local,  1, MPI_INT, 0, 1, MPI_COMM_WORLD);
        MPI_Send(A->row_ptr, A->rows_local+1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(A->col_idx, A->nnz_local,    MPI_INT, 0, 3, MPI_COMM_WORLD);
        MPI_Send(A->value,   A->nnz_local, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD);
        if(first_time)
        {
            B->rank = rank;
            B->nprocs = nprocs;
            B->rows = rows;
            B->cols = cols;
            MPI_Recv(&B->nnz,         1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
            MPI_Recv(&B->rows_local,  1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(&B->rows_offset, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(&B->nnz_local,   1, MPI_INT, 0, 3, MPI_COMM_WORLD, &status);
            B->row_ptr = (int*)malloc((B->rows_local+1)*sizeof(int));
            B->col_idx = (int*)malloc(B->nnz_local*sizeof(int));
            B->value = (double*)malloc(B->nnz_local*sizeof(double));
        }
        MPI_Recv(B->row_ptr, B->rows_local+1, MPI_INT,    0, 4, MPI_COMM_WORLD, &status);
        MPI_Recv(B->col_idx, B->nnz_local,    MPI_INT,    0, 5, MPI_COMM_WORLD, &status);
        MPI_Recv(B->value,   B->nnz_local,    MPI_DOUBLE, 0, 6, MPI_COMM_WORLD, &status);
    }
    return 0;
}

/* B = AT*A */
int sparse_ATA_type0(CSR *B, CSR *A, bool first_time)
{
    if(first_time) /* The first time to use Matrix A, you can do some pre-process. */
    {
    }
    int rows =  A->rows;
    int cols =  A->cols;
    int nnz  =  A->nnz;
    int rank = A->rank;
    int nprocs = A->nprocs;

    if(rank == 0)
    {
        int *rows_per_proc = (int*)malloc(nprocs*sizeof(int));
        int *row_ptr  = (int*)malloc((rows+1)*sizeof(int));
        int *pointerE = NULL;//(int*)malloc(rows*sizeof(int));
        int *col_idx  = (int*)malloc(nnz*sizeof(int));
        double *value = (double*)malloc(nnz*sizeof(double));
        memcpy(row_ptr, A->row_ptr, (A->rows_local+1)*sizeof(int));
        memcpy(col_idx, A->col_idx, A->nnz_local*sizeof(int));
        memcpy(value,   A->value,   A->nnz_local*sizeof(double));
        int rows_offset = A->rows_local;
        int nnz_offset = A->nnz_local;
        rows_per_proc[0] = A->rows_local;
        for(int i = 1; i < nprocs; i ++)
        {
            int rows_local;
            int nnz_local;
            MPI_Status status;
            MPI_Recv(&rows_local, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
            MPI_Recv(&nnz_local,  1, MPI_INT, i, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(row_ptr + rows_offset, rows_local+1, MPI_INT, i, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(col_idx + nnz_offset,  nnz_local,    MPI_INT, i, 3, MPI_COMM_WORLD, &status);
            MPI_Recv(value   + nnz_offset,  nnz_local, MPI_DOUBLE, i, 4, MPI_COMM_WORLD, &status);       
            rows_offset += rows_local;
            nnz_offset += nnz_local;
            rows_per_proc[i] = rows_local;
        }

        CSR a;
        a.rows = rows;
        a.cols = cols;
        a.nnz = nnz;
        a.row_ptr = row_ptr;
        a.col_idx = col_idx;
        a.value = value;
        CSR at;
        transpose(&a, &at);
        CSR c;
        multi(&at, &at, &c, first_time);

        free(row_ptr); row_ptr = NULL;
        free(col_idx); col_idx = NULL;
        free(value);   value   = NULL;

        rows = c.rows;
        cols = c.cols;
        row_ptr = c.row_ptr;
        col_idx = c.col_idx;
        value = c.value;

        /*{
          int i, j;
          for ( i = 0; i < rows; i ++ ) {
          Qsort(col_idx, value, row_ptr[i], row_ptr[i+1]-1);
          }
          }*/

        if(first_time)
        {
            B->rank = rank;
            B->nprocs = nprocs;
            B->rows = rows;
            B->cols = cols;
            B->nnz = row_ptr[rows] - row_ptr[0];
            B->rows_local  = rows_per_proc[0];
            B->rows_offset = 0;
            B->nnz_local   = row_ptr[B->rows_local] - row_ptr[0];
            B->row_ptr = (int*)malloc((B->rows_local+1)*sizeof(int));
            B->col_idx = (int*)malloc(B->nnz_local*sizeof(int));
            B->value = (double*)malloc(B->nnz_local*sizeof(double));
            rows_offset = B->rows_local;
            int total_nnz = row_ptr[rows] - row_ptr[0];
            for(int i = 1; i < nprocs; i ++)
            {
                int rows_local = rows_per_proc[i];
                int nnz_local = row_ptr[rows_offset+rows_local] - row_ptr[rows_offset];
                MPI_Send(&total_nnz, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                MPI_Send(&rows_local, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
                MPI_Send(&rows_offset, 1, MPI_INT, i, 2, MPI_COMM_WORLD);
                MPI_Send(&nnz_local, 1, MPI_INT, i, 3, MPI_COMM_WORLD);
                rows_offset += rows_local;
            }
        }
        memcpy(B->row_ptr, row_ptr, (rows_per_proc[0]+1)*sizeof(int));
        memcpy(B->col_idx, col_idx, (row_ptr[B->rows_local]-row_ptr[0])*sizeof(int));
        memcpy(B->value,   value,   (row_ptr[B->rows_local]-row_ptr[0])*sizeof(double));
        rows_offset = B->rows_local;
        for(int i = 1; i < nprocs; i ++)
        {
            int rows_local = rows_per_proc[i];
            MPI_Send(row_ptr + rows_offset, rows_local+1, MPI_INT, i, 4, MPI_COMM_WORLD);
            MPI_Send(col_idx + row_ptr[rows_offset], row_ptr[rows_offset+rows_local]-row_ptr[rows_offset], MPI_INT,    i, 5, MPI_COMM_WORLD);
            MPI_Send(value   + row_ptr[rows_offset], row_ptr[rows_offset+rows_local]-row_ptr[rows_offset], MPI_DOUBLE, i, 6, MPI_COMM_WORLD);
            rows_offset += rows_local;
        }

        free(rows_per_proc);
    }
    else
    {
        MPI_Status status;
        MPI_Send(&A->rows_local, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&A->nnz_local,  1, MPI_INT, 0, 1, MPI_COMM_WORLD);
        MPI_Send(A->row_ptr, A->rows_local+1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(A->col_idx, A->nnz_local,    MPI_INT, 0, 3, MPI_COMM_WORLD);
        MPI_Send(A->value,   A->nnz_local, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD);
        if(first_time)
        {
            B->rank = rank;
            B->nprocs = nprocs;
            B->rows = rows;
            B->cols = cols;
            MPI_Recv(&B->nnz,         1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
            MPI_Recv(&B->rows_local,  1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(&B->rows_offset, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(&B->nnz_local,   1, MPI_INT, 0, 3, MPI_COMM_WORLD, &status);
            B->row_ptr = (int*)malloc((B->rows_local+1)*sizeof(int));
            B->col_idx = (int*)malloc(B->nnz_local*sizeof(int));
            B->value = (double*)malloc(B->nnz_local*sizeof(double));
        }
        MPI_Recv(B->row_ptr, B->rows_local+1, MPI_INT,    0, 4, MPI_COMM_WORLD, &status);
        MPI_Recv(B->col_idx, B->nnz_local,    MPI_INT,    0, 5, MPI_COMM_WORLD, &status);
        MPI_Recv(B->value,   B->nnz_local,    MPI_DOUBLE, 0, 6, MPI_COMM_WORLD, &status);
    }
    return 0;
}

/* B = AT*A */
int sparse_ATA_type1(CSR *B, CSR *A, bool first_time)
{
    if(first_time) /* The first time to use Matrix A, you can do some pre-process. */
    {
    }
    int rows =  A->rows;
    int cols =  A->cols;
    int nnz  =  A->nnz;
    int rank = A->rank;
    int nprocs = A->nprocs;

    if(rank == 0)
    {
        int *rows_per_proc = (int*)malloc(nprocs*sizeof(int));
        int *row_ptr  = (int*)malloc((rows+1)*sizeof(int));
        int *pointerE = NULL;//(int*)malloc(rows*sizeof(int));
        int *col_idx  = (int*)malloc(nnz*sizeof(int));
        double *value = (double*)malloc(nnz*sizeof(double));
        memcpy(row_ptr, A->row_ptr, (A->rows_local+1)*sizeof(int));
        memcpy(col_idx, A->col_idx, A->nnz_local*sizeof(int));
        memcpy(value,   A->value,   A->nnz_local*sizeof(double));
        int rows_offset = A->rows_local;
        int nnz_offset = A->nnz_local;
        rows_per_proc[0] = A->rows_local;
        for(int i = 1; i < nprocs; i ++)
        {
            int rows_local;
            int nnz_local;
            MPI_Status status;
            MPI_Recv(&rows_local, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
            MPI_Recv(&nnz_local,  1, MPI_INT, i, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(row_ptr + rows_offset, rows_local+1, MPI_INT, i, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(col_idx + nnz_offset,  nnz_local,    MPI_INT, i, 3, MPI_COMM_WORLD, &status);
            MPI_Recv(value   + nnz_offset,  nnz_local, MPI_DOUBLE, i, 4, MPI_COMM_WORLD, &status);       
            rows_offset += rows_local;
            nnz_offset += nnz_local;
            rows_per_proc[i] = rows_local;
        }

        CSR a;
        a.rows = rows;
        a.cols = cols;
        a.nnz = nnz;
        a.row_ptr = row_ptr;
        a.col_idx = col_idx;
        a.value = value;
        CSR at;
        transpose(&a, &at);
        CSR c;
        multi(&at, &at, &c, first_time);

        free(row_ptr); row_ptr = NULL;
        free(col_idx); col_idx = NULL;
        free(value);   value   = NULL;

        rows = c.rows;
        cols = c.cols;
        row_ptr = c.row_ptr;
        col_idx = c.col_idx;
        value = c.value;

        if(first_time)
        {
            B->rank = rank;
            B->nprocs = nprocs;
            B->rows = rows;
            B->cols = cols;
            B->nnz = row_ptr[rows] - row_ptr[0];
            B->rows_local  = rows_per_proc[0];
            B->rows_offset = 0;
            B->nnz_local   = row_ptr[B->rows_local] - row_ptr[0];
            B->row_ptr = (int*)malloc((B->rows_local+1)*sizeof(int));
            B->col_idx = (int*)malloc(B->nnz_local*sizeof(int));
            B->value = (double*)malloc(B->nnz_local*sizeof(double));
            rows_offset = B->rows_local;
            int total_nnz = row_ptr[rows] - row_ptr[0];
            for(int i = 1; i < nprocs; i ++)
            {
                int rows_local = rows_per_proc[i];
                int nnz_local = row_ptr[rows_offset+rows_local] - row_ptr[rows_offset];
                MPI_Send(&total_nnz, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                MPI_Send(&rows_local, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
                MPI_Send(&rows_offset, 1, MPI_INT, i, 2, MPI_COMM_WORLD);
                MPI_Send(&nnz_local, 1, MPI_INT, i, 3, MPI_COMM_WORLD);
                rows_offset += rows_local;
            }
        }
        memcpy(B->row_ptr, row_ptr, (rows_per_proc[0]+1)*sizeof(int));
        memcpy(B->col_idx, col_idx, (row_ptr[B->rows_local]-row_ptr[0])*sizeof(int));
        memcpy(B->value,   value,   (row_ptr[B->rows_local]-row_ptr[0])*sizeof(double));
        rows_offset = B->rows_local;
        for(int i = 1; i < nprocs; i ++)
        {
            int rows_local = rows_per_proc[i];
            MPI_Send(row_ptr + rows_offset, rows_local+1, MPI_INT, i, 4, MPI_COMM_WORLD);
            MPI_Send(col_idx + row_ptr[rows_offset], row_ptr[rows_offset+rows_local]-row_ptr[rows_offset], MPI_INT,    i, 5, MPI_COMM_WORLD);
            MPI_Send(value   + row_ptr[rows_offset], row_ptr[rows_offset+rows_local]-row_ptr[rows_offset], MPI_DOUBLE, i, 6, MPI_COMM_WORLD);
            rows_offset += rows_local;
        }

        free(rows_per_proc);
    }
    else
    {
        MPI_Status status;
        MPI_Send(&A->rows_local, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&A->nnz_local,  1, MPI_INT, 0, 1, MPI_COMM_WORLD);
        MPI_Send(A->row_ptr, A->rows_local+1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(A->col_idx, A->nnz_local,    MPI_INT, 0, 3, MPI_COMM_WORLD);
        MPI_Send(A->value,   A->nnz_local, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD);
        if(first_time)
        {
            B->rank = rank;
            B->nprocs = nprocs;
            B->rows = rows;
            B->cols = cols;
            MPI_Recv(&B->nnz,         1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
            MPI_Recv(&B->rows_local,  1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(&B->rows_offset, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(&B->nnz_local,   1, MPI_INT, 0, 3, MPI_COMM_WORLD, &status);
            B->row_ptr = (int*)malloc((B->rows_local+1)*sizeof(int));
            B->col_idx = (int*)malloc(B->nnz_local*sizeof(int));
            B->value = (double*)malloc(B->nnz_local*sizeof(double));
        }
        MPI_Recv(B->row_ptr, B->rows_local+1, MPI_INT,    0, 4, MPI_COMM_WORLD, &status);
        MPI_Recv(B->col_idx, B->nnz_local,    MPI_INT,    0, 5, MPI_COMM_WORLD, &status);
        MPI_Recv(B->value,   B->nnz_local,    MPI_DOUBLE, 0, 6, MPI_COMM_WORLD, &status);
    }
    return 0;
}

