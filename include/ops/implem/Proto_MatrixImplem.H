
template<typename T>
Matrix<T>::Matrix(unsigned int a_numRows, unsigned int a_numCols)
{
    define(a_numRows, a_numCols);
}

template<typename T>
Matrix<T>::Matrix(std::shared_ptr<T>& a_buffer, unsigned int a_numRows, unsigned int a_numCols)
{
    PR_TIME("Matrix(buffer, numRows, numCols)");
    bli_obj_create_without_buffer(blisDatatype(), a_numRows, a_numCols, &m_object);
    bli_obj_attach_buffer(a_buffer.get(), 1, a_numRows, 0, &m_object);
    m_alias = true;
}

template<typename T>
Matrix<T>::Matrix(std::initializer_list<std::initializer_list<T>> a_args)
{
    PR_TIME("Matrix(initializer_list)");
    int M = a_args.size();
    int N = (*a_args.begin()).size();
    define(M,N);
    int index = 0;
    int ii = 0;
    int jj = 0;
    for (auto row : a_args)
    {
        PROTO_ASSERT(row.size() == N,
                "Matrix(initializer_list) | Error: inconsistent number of columns specified");
        jj = 0;
        for (auto val : row)
        {
            set(ii, jj, val);
            jj++;
        }
        ii++;
    }
}

template<typename T>
Matrix<T>::Matrix(const Matrix<T>&& a_matrix)
{
    PR_TIME("Matrix(Matrix&&)");
    m_alias = a_matrix.m_alias;
    m_object = std::move(a_matrix.m_object);
}

template<typename T>
Matrix<T>::~Matrix()
{
    PR_TIME("~Matrix()");
    if (!m_alias)
    {
        bli_obj_free(&m_object);
    }
}

template<typename T>
void Matrix<T>::define(unsigned int a_numRows, unsigned int a_numCols)
{
    PR_TIME("Matrix(numRows, numCols)");
    // using column-major order and suppressing BLIS' buffer packing
    bli_obj_create(blisDatatype(), a_numRows, a_numCols, 1, a_numRows, &m_object);
    m_alias = false;
}


template<typename T>
Matrix<T>&
Matrix<T>::operator=(const Matrix<T>&& a_matrix)
{
    PR_TIME("Matrix::operator=(Matrix&&)");
    m_alias = a_matrix.m_alias;
    m_object = std::move(a_matrix.m_object);
    return *this;
}

template<typename T>
T 
Matrix<T>::get(unsigned int a_row, unsigned int a_col) const
{
    PR_TIME("Matrix::get(row, col)");
    PROTO_ASSERT(a_row < numRows(), 
            "Matrix::get | Error: row index %u is out of bounds (numRows() = %u)",
            a_row, numRows());
    PROTO_ASSERT(a_col < numCols(), 
            "Matrix::get | Error: col index %u is out of bounds (numCols() = %u)",
            a_col, numCols());
    T real;
    T imag; //unused
    //TODO: make this work properly if T is complex
    switch (bli_obj_conjtrans_status(&m_object))
    {
        case BLIS_TRANSPOSE:
            bli_getijm(a_col, a_row, &m_object, &real, &imag); break;
        case BLIS_NO_TRANSPOSE:
        default:
            bli_getijm(a_row, a_col, &m_object, &real, &imag); break;
    }
    return real;
}

template<typename T>
T
Matrix<T>::get(unsigned int a_index) const
{
    PROTO_ASSERT(a_index < numRows()*numCols(),
            "Matrix::get | Error: index %u is out of bounds", a_index);
    unsigned int ii = a_index % numRows();
    unsigned int jj = a_index / numRows();
    return get(ii, jj);
}

template<typename T>
void 
Matrix<T>::set(unsigned int a_row, unsigned int a_col, const T& a_val)
{
    PR_TIME("Matrix::set(row, col, val)");
    PROTO_ASSERT(a_row < numRows(), 
            "Matrix::get | Error: row index %u is out of bounds (numRows() = %u)",
            a_row, numRows());
    PROTO_ASSERT(a_col < numCols(), 
            "Matrix::get | Error: col index %u is out of bounds (numCols() = %u)",
            a_col, numCols());
    //TODO: make this work properly if T is complex
    switch (bli_obj_conjtrans_status(&m_object))
    {
        case BLIS_TRANSPOSE:
            bli_setijm(a_val, 0, a_col, a_row, &m_object);
        case BLIS_NO_TRANSPOSE:
        default:
            bli_setijm(a_val, 0, a_row, a_col, &m_object);
    }
}

template<typename T>
void 
Matrix<T>::set(unsigned int a_index, const T& a_val)
{
    PROTO_ASSERT(a_index < numRows()*numCols(),
            "Matrix::get | Error: index %u is out of bounds", a_index);
    unsigned int ii = a_index % numRows();
    unsigned int jj = a_index / numRows();
    return set(ii, jj, a_val);
}

template<typename T>
void 
Matrix<T>::set(const T& a_val)
{
    PR_TIME("Matrix::set(val)");
    if (a_val == 0)
    {
        bli_setm(&BLIS_ZERO, &m_object);
    } else if (a_val == 1)
    {
        bli_setm(&BLIS_ONE, &m_object);
    } else if (a_val == -1)
    {
        bli_setm(&BLIS_MINUS_ONE, &m_object);
    } else {
        obj_t alpha;
        bli_obj_create_1x1(blisDatatype(), &alpha);
        bli_setsc(a_val, 0, &alpha);
        bli_setm(&alpha, &m_object);
    }
}

template<typename T>
bool
Matrix<T>::operator==(const Matrix<T>& a_rhs) const
{
    if (numRows() != a_rhs.numRows()) { return false; }
    if (numCols() != a_rhs.numCols()) { return false; }

    for (int ii = 0; ii < M(); ii++)
    {
        for (int jj = 0; jj < N(); jj++)
        {
            if (get(ii,jj) != a_rhs.get(ii,jj)) { return false; }
        }
    }
    return true;
}

template<typename T>
bool
Matrix<T>::operator!=(const Matrix<T>& a_rhs) const
{
    return !((*this) == a_rhs);
}

template<typename T>
unsigned int
Matrix<T>::numRows() const
{
    PR_TIME("Matrix::numRows");
    return bli_obj_length_after_trans(&m_object);
}

template<typename T>
unsigned int
Matrix<T>::numCols() const
{
    PR_TIME("Matrix::numCols");
    return bli_obj_width_after_trans(&m_object);
}

template<typename T>
Matrix<T>
Matrix<T>::slice(
        unsigned int a_numRows, unsigned int a_numCols,
        unsigned int a_rowOffset, unsigned int a_colOffset)
{
    PR_TIME("Matrix::slice");
    PROTO_ASSERT(a_rowOffset + a_numRows <= numRows(),
            "Matrix::slice | Error: row index is out of bounds");
    PROTO_ASSERT(a_colOffset + a_numCols <= numCols(),
            "Matrix::slice | Error: column index is out of bounds");
    Matrix<T> ret;
    bli_acquire_mpart(
            a_rowOffset, a_colOffset,
            a_numRows, a_numCols,
            &m_object, &(ret.m_object));
    ret.m_alias = true;
    return ret;
}

template<typename T>
const Matrix<T>
Matrix<T>::slice(
        unsigned int a_numRows, unsigned int a_numCols,
        unsigned int a_rowOffset, unsigned int a_colOffset) const
{
    PR_TIME("Matrix::slice");
    PROTO_ASSERT(a_rowOffset + a_numRows <= numRows(),
            "Matrix::slice | Error: row index is out of bounds");
    PROTO_ASSERT(a_colOffset + a_numCols <= numCols(),
            "Matrix::slice | Error: column index is out of bounds");
    Matrix<T> ret;
    bli_acquire_mpart(
            a_rowOffset, a_colOffset,
            a_numRows, a_numCols,
            &m_object, &(ret.m_object));
    ret.m_alias = true;
    return ret;
}

template<typename T>
Matrix<T>
Matrix<T>::row(unsigned int a_rowIndex)
{
    PROTO_ASSERT(a_rowIndex < numRows(),
            "Matrix::row | Error: row index %u is out of bounds (numRows() = %u",
            a_rowIndex, numRows());
    return slice(1, numCols(), a_rowIndex, 0);
}

template<typename T>
Matrix<T>
Matrix<T>::col(unsigned int a_colIndex)
{
    PROTO_ASSERT(a_colIndex < numCols(),
            "Matrix::col | Error: col index %u is out of bounds (numCols() = %u",
            a_colIndex, numCols());
    return slice(numRows(), 1, 0, a_colIndex);
}

template<typename T>
void
Matrix<T>::copyTo(Matrix<T>& a_rhs) const
{
    PROTO_ASSERT(this->M() == a_rhs.M() && this->N() == a_rhs.N(),
            "Matrix::copyTo | Error: Matrices must be the same size)");
    bli_copym(&m_object, &(a_rhs.m_object));
}

template<typename T>
Matrix<T>
Matrix<T>::operator+(const Matrix<T>& a_rhs) const
{
    PROTO_ASSERT(this->M() == a_rhs.M() && this->N() == a_rhs.N(),
            "Matrix::operator+ | Error: Matrices must be the same size)");
    Matrix<T> ret(M(), N());
    this->copyTo(ret);
    bli_addm(&(a_rhs.m_object), &(ret.m_object));
    return ret;
}

template<typename T>
void
Matrix<T>::operator+=(const Matrix<T>& a_rhs)
{
    PROTO_ASSERT(this->M() == a_rhs.M() && this->N() == a_rhs.N(),
            "Matrix::operator+ | Error: Matrices must be the same size)");
    bli_addm(&(a_rhs.m_object), &m_object);
}

template<typename T>
Matrix<T>
Matrix<T>::operator+(T a_scale) const
{
    Matrix<T> ret(M(), N());
    ret.set(a_scale);
    ret += (*this);
    return ret;
}

template<typename T>
void
Matrix<T>::operator+=(T a_scale)
{
    Matrix<T> tmp(M(), N());
    tmp.set(a_scale);
    (*this) += tmp;
}

template<typename T>
Matrix<T>
Matrix<T>::operator-(const Matrix<T>& a_rhs) const
{
    PROTO_ASSERT(this->M() == a_rhs.M() && this->N() == a_rhs.N(),
            "Matrix::operator+ | Error: Matrices must be the same size)");
    Matrix<T> ret(M(), N());
    this->copyTo(ret);
    bli_subm(&(a_rhs.m_object), &(ret.m_object));
    return ret;
}

template<typename T>
void
Matrix<T>::operator-=(const Matrix<T>& a_rhs)
{
    PROTO_ASSERT(this->M() == a_rhs.M() && this->N() == a_rhs.N(),
            "Matrix::operator+ | Error: Matrices must be the same size)");
    bli_subm(&(a_rhs.m_object), &m_object);
}

template<typename T>
Matrix<T>
Matrix<T>::operator-(T a_scale) const
{
    return (*this) + (-a_scale);
}

template<typename T>
void
Matrix<T>::operator-=(T a_scale)
{
    (*this) += (-a_scale);
}

template<typename T>
Matrix<T>
Matrix<T>::operator*(const Matrix<T>& a_rhs) const
{
    PROTO_ASSERT(N() == a_rhs.M(),
            "Matrix::operator* | Error: Incompatible Matrix dimensions.");
    Matrix<T> ret(M(), a_rhs.N());
    if (a_rhs.isVector())
    {
        bli_gemv(&BLIS_ONE, &m_object, &(a_rhs.m_object), &BLIS_ZERO, &(ret.m_object));
    } else {
        bli_gemm(&BLIS_ONE, &m_object, &(a_rhs.m_object), &BLIS_ZERO, &(ret.m_object));
    }
    return ret;
}

template<typename T>
Matrix<T>
Matrix<T>::operator*(T a_scale) const
{
    Matrix<T> ret(M(), N());
    if (a_scale == 0)
    {
        ret.set(0);
    } else if (a_scale == 1)
    {
        this->copyTo(ret);
    } else {
        obj_t alpha;
        bli_obj_create_1x1(blisDatatype(), &alpha);
        bli_setsc(a_scale, 0, &alpha);
        bli_scal2m(&alpha, &m_object, &(ret.m_object));
    }
    return ret;
}

template<typename T>
void
Matrix<T>::operator*=(T a_scale)
{
    if (a_scale == 0)
    {
        this->set(0);
    } else if (a_scale == 1)
    {
        return;
    } else {
        obj_t alpha;
        bli_obj_create_1x1(blisDatatype(), &alpha);
        bli_setsc(a_scale, 0, &alpha);
        bli_scalm(&alpha, &m_object);
    }
}

template<typename T>
T*
Matrix<T>::data()
{
    return (T*)bli_obj_buffer(&m_object);
}

template<typename T>
const T*
Matrix<T>::data() const
{
    return (T*)bli_obj_buffer(&m_object);
}

template<typename T>
const Matrix<T>
Matrix<T>::transpose() const
{
    const Matrix<T> ret = slice(M(), N(), 0, 0);
    Matrix<T>* mut = const_cast<Matrix<T>*>(&ret);
    bli_obj_toggle_trans(&(mut->m_object));
    return ret;
}

template<typename T>
Matrix<T>
Matrix<T>::inverse() const
{
    PROTO_ASSERT(M() == N(),
            "Matrix::inverse | Error: Matrix is not square");
    Matrix<T> ret(M(), N());
    int n = M();
    int nn = n*n;
    const T* srcBuffer = data();
    T* dstBuffer = ret.data();
    proto_memcpy<HOST, HOST>(srcBuffer, dstBuffer, nn*sizeof(T));
    int* pivots = (int*)proto_malloc<HOST>(n*sizeof(int));
    T* work = (T*)proto_malloc<HOST>(nn*sizeof(T));
    int info;
    PROTO_LAPACK(GETRF,getrf)(&n, &n, dstBuffer, &n, pivots, &info);
    PROTO_ASSERT(info <= 0,
            "Matrix::inverse | Error: Matrix LU factorization failed. Matrix may be singular.");
    PROTO_LAPACK(GETRI,getri)(&n, dstBuffer, &n, pivots, work, &nn, &info);
    PROTO_ASSERT(info <= 0,
            "Matrix::inverse | Error: Matrix inversion failed. Matrix may be singular.");
    proto_free<HOST>(work);
    proto_free<HOST>(pivots);
    return ret;
}

template<typename T>
unsigned int
Matrix<T>::size() const
{
    return M()*N();
}

template<typename T>
size_t
Matrix<T>::linearSize() const
{
    return size()*sizeof(T);
}

template<typename T>
bool 
Matrix<T>::isVector() const
{
    return ((numCols() == 1) || (numRows() == 1));
}

template<typename T>
bool 
Matrix<T>::isScalar() const
{
    return ((numCols() == 1) && (numRows() == 1));
}

template<typename T>
void 
Matrix<T>::print(std::string a_format) const
{
    Matrix<T> A(M(), N());
    this->copyTo(A); //this make sure transposed matrices print correctly
    if (a_format == "")
    {
        switch (blisDatatype())
        {
            case BLIS_DOUBLE:
                bli_printm("", &(A.m_object), "%4.1f", "");
                break;
            default:
                bli_printm("", &(A.m_object), "%i", "");
                break;

        }
    } else {
        bli_printm("", &m_object, a_format.c_str(), "");
    }
}

template<typename T>
double
Matrix<T>::conditionNumber() const
{
}

template<typename T>
void
Matrix<T>::checkConditionNumber() const
{
    double inverse = 1.0/conditionNumber();
    double small = 1.0e-6;
    double reallysmall = 1.0e-15;
    if(inverse < reallysmall)
    {
        pout() << "matrix is poorly conditioned: 1/condition_number = " << inverse << endl;
    }
    else if(inverse < small)
    {
        pout() << "matrix is marginally conditioned: 1/condition_number = " << inverse << endl;
    }
    else
    {
        pout() << "matrix might be OK: 1/condition_number = " << inverse << endl;
    }
}

template<typename T>
void solveLS(Matrix<T>& A, Matrix<T>& B)
{
    PR_TIME("solveLeastSquares");
    // TODO - check that the sizes of A, B and C are compatible
    int M = A.M();
    int N = A.N();
    int NRHS = B.N();
    int LDA = M;
    int LDB = std::max(M,N);
    PROTO_ASSERT(B.M() == M,
            "solveLS | Error: Incompatible matrix dimensions. A.M() == %u, b.M() = %u.",
            A.M(), B.M());

    int LWORK[2] = {1,1};
    LWORK[0] = 2*M*N;

    Matrix<T> WORK(2*M*N, 1);
    WORK.set(0.0);

    char TRANS = 'N';
    int INFO;

    PROTO_LAPACK(GELS,gels)(&TRANS, &M, &N, &NRHS, A.data(), &LDA, 
            B.data(), &LDB, WORK.data(), LWORK, &INFO);

    PROTO_ASSERT(INFO == 0,
            "solveLS | Error: GELS returned an error flag. Matrix may be singular.");
}

template<typename T>
void solveRRLS(Matrix<T>& A, Matrix<T>& b)
{
    // - check that the sizes of A, b are compatible
    int M = A.M();
    int N = A.N();
    int NRHS =  b.N();
    PROTO_ASSERT(b.M() == M,
            "solveRRLS | Error: Incompatible matrix domensions. A.M() = %u, b.M() = %u.",
            A.M(), b.M());
    PROTO_ASSERT(M >= N, "solveRRLS | Error: System is not over-determined.");

    // - Compute A transpose
    auto At = A.transpose(); 

    // - Calculate QR factorization of A' with DGEQRF
    int LWORK[2] = {1,1};
    LWORK[0] = 2*M*N;
    Matrix<T> WORK(LWORK[0], 1);
    WORK.setVal(0.0);

    Matrix<T> TAU(N, 1);
    TAU.setVal(0.0);

    int INFO;
    PROTO_LAPACK(GEQRF,geqrf)(&N, &M, At.dataPtr(), &N,
            TAU.dataPtr(), WORK.dataPtr(), LWORK, &INFO);
    
    PROTO_ASSERT(INFO == 0,
            "solveRRLS | Error: GEQRF returned an error flag. Matrix may be singular.");

    char SIDE = 'L';
    char TRANS = 'T';
    PROTO_LAPACK(ORMQR,ormqr)(&SIDE, &TRANS, &N, &NRHS, &N, 
            At.data(), &N, TAU.data(), b.data(), &N,
            WORK.data(), LWORK, &INFO);

    PROTO_ASSERT(INFO == 0,
            "solveRRLS | Error: ORMQR returned an error flag. Matrix may be singular.");

    // - Solve R x = (Q' * b) using DTRTRS
    char UPLO = 'U';
    TRANS = 'N';
    char DIAG = 'N';
    PROTO_LAPACK(TRTRS,trtrs)(&UPLO, &TRANS, &DIAG, &N, &NRHS, 
            At.data(), &N, b.data(), &N, &INFO);
    
    PROTO_ASSERT(INFO == 0,
            "solveRRLS | Error: TRTRS returned an error flag. Matrix may be singular.");
}
