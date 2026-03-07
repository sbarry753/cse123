#pragma once
#include <cstddef>
#include <random>
#include <stddef.h>
#include "arm_math.h"

class Matrix2D {
    public:
        Matrix2D(size_t rows, size_t cols) : rows_(rows), cols_(cols) {
            data_ = new float*[rows_];
            for (size_t i = 0; i < rows_; i++) {
                data_[i] = new float[cols]();
            }
        }

        ~Matrix2D() {
            for (size_t i = 0; i < rows_; i++) {
                delete[] data_[i];
            }
            
            delete[] data_;
        }

        void fill_matrix() {
            std::mt19937 rng(std::random_device{}());
            std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

            for (size_t i = 0; i < rows_; i++) {
                for (size_t j = 0; j < cols_; j++) {
                    data_[i][j] = dist(rng);
                }
            }
        }

        inline float checksum() {
            return data_[0][0] +data_ [rows_ - 1][cols_ -1];
        }

        static inline void reg_matmul(const Matrix2D& A, const Matrix2D& B, Matrix2D& C) {
            for (size_t i = 0; i < A.rows_; i++) {
                for (size_t j = 0; j < B.cols_; j++) {
                    float sum = 0.0f;
                    for (size_t k = 0; k < A.cols_; k++) {
                        sum += A.data_[i][k] * B.data_[k][j];
                    }
                    C.data_[i][j] += sum;
                }
            }
        }

        static inline void better_matmul(const Matrix2D& A, const Matrix2D& B, Matrix2D& C) {
            for (size_t i = 0; i < A.rows_; i++) {
                for (size_t k = 0; k < A.cols_; k++) {
                    const float a_ik = A.data_[i][k];
                    for (size_t j = 0; j < B.cols_; j++) {
                        C.data_[i][j] += a_ik * B.data_[k][j];
                    }
                }
            }
        }

        static inline void tiled_matmul(const Matrix2D& A, const Matrix2D& B, Matrix2D& C) {
            size_t tile_size = 8;

            for (size_t kk = 0; kk < A.cols_; kk += tile_size) {
                for (size_t i = 0; i < A.rows_; i++) {
                    size_t tile_end = std::min(A.cols_, kk + tile_size);

                    for (size_t k = kk; k < tile_end; k++)  {
                        float a_val = A.data_[i][k];

                        for (size_t j = 0; j < B.cols_; j++) {
                            C.data_[i][j] += a_val * B.data_[k][j];
                        }
                    }
                }
            }
        }
    private:
        size_t rows_;
        size_t cols_;
        float **data_;

};

class Matrix1D {
    public:
        Matrix1D(size_t rows, size_t cols) : rows_(rows), cols_(cols) {
            data_ = new float[rows_ * cols_]();
        }

        ~Matrix1D() {
            delete[] data_;
        }

        void fill_matrix() {
            std::mt19937 rng(std::random_device{}());
            std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

            for (size_t i = 0; i < rows_ * cols_; i++) {
                data_[i] = dist(rng);
            }
        }

        inline float checksum() {
            return data_[0] + data_[rows_ * cols_ -1];
        }

        static inline void reg_matmul(const Matrix1D& A, const Matrix1D& B, Matrix1D& C) {
            for (size_t i = 0; i < A.rows_; i++) {
                for (size_t j = 0; j < B.cols_; j++) {
                    float sum = 0.0f;
                    for (size_t k = 0; k < A.cols_; k++) {
                        sum += A.data_[i * A.cols_ + k] * B.data_[k * B.cols_ + j];
                    }
                    C.data_[i * C.cols_ + j] += sum;
                }
            }
        }

        static inline void better_matmul(const Matrix1D& A, const Matrix1D& B, Matrix1D& C) {
            for (size_t i = 0; i < A.rows_; i++) {
                for (size_t k = 0; k < A.cols_; k++) {
                    const float a_ik = A.data_[i * A.cols_ + k];
                    for (size_t j = 0; j < B.cols_; j++) {
                        C.data_[i * C.cols_ + j] += a_ik * B.data_[k * B.cols_ + j];
                    }
                }
            }
        }

        static inline void rpc_matmul(const Matrix1D& A, const Matrix1D& B, Matrix1D& C) {
            for (size_t i = 0; i < A.rows_; i++) {
                float *c_row = C.data_ + i*C.cols_;

                for (size_t k = 0; k < A.cols_; k++) {
                    const float a_ik = A.data_[i * A.cols_ + k];
                    float *b_row = B.data_ + k*B.cols_;

                    for (size_t j = 0; j < B.cols_; j++) {
                        c_row[j] += a_ik * b_row[j];
                    }
                }
            }
        }

        static inline void rpc_reg_matmul(const Matrix1D& A, const Matrix1D& B, const Matrix1D& C) {
            for (size_t i = 0; i < A.rows_; i++) {
                for (size_t j = 0; j < B.cols_; j++) {
                    float sum = 0.0f;
                    float *a_row = A.data_ + i*A.cols_;

                    for (size_t k = 0; k < A.cols_; k++) {
                        sum += a_row[k] * B.data_[k * B.cols_ + j];
                    }
                    C.data_[i * C.cols_ + j] += sum;
                }
            }
        }

        static inline void tiled_matmul(const Matrix1D& A, const Matrix1D& B, Matrix1D& C) {
            size_t tile_size = 8;

            for (size_t kk = 0; kk < A.cols_; kk += tile_size) {
                for (size_t i = 0; i < A.rows_; i++) {
                    size_t tile_end = std::min(A.cols_, kk + tile_size);

                    for (size_t k = kk; k < tile_end; k++) {
                        float a_val = A.data_[i * A.cols_ + k];

                        for (size_t j = 0; j < B.cols_; j++) {
                            C.data_[i * C.cols_ + j] += a_val * B.data_[k * B.cols_ + j];
                        }
                    }
                }
            }
        }


    private:
        size_t rows_;
        size_t cols_;
        float *data_;
};

class CMSISMatrix {
    public:
        CMSISMatrix(size_t rows, size_t cols, float *data) : rows_(rows), cols_(cols) {
            arm_mat_init_f32(&mat_, static_cast<uint16_t>(rows_), static_cast<uint16_t>(cols_), data);
        }

        void fill_matrix() {
            std::mt19937 rng(std::random_device{}());
            std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

            for (size_t i = 0; i < rows_ * cols_; i++) {
                mat_.pData[i] = dist(rng);
            }
        }

        inline float checksum() const {
            return mat_.pData[0] + mat_.pData[rows_ * cols_ - 1];
        }

        static inline void matmul(const CMSISMatrix& A, const CMSISMatrix& B, CMSISMatrix& C) {
            arm_mat_mult_f32(&A.mat_, &B.mat_, &C.mat_);
        }


    private:
        size_t rows_;
        size_t cols_;
        arm_matrix_instance_f32 mat_;
}; 