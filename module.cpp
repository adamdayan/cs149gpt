#include <torch/extension.h>
#include <ATen/ATen.h>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <vector>
#include <immintrin.h>

// Uncomment for ISPC
//#include "module_ispc.h"
//using namespace ispc;

// ------------------------------------ //
// 	WARM-UP: ACCESSING TENSORS      //
// ------------------------------------ //

// Step #1: Understand Read/Write Accessors for a 2D Tensor
inline float twoDimRead(std::vector<float> &tensor, int &x, int &y, const int &sizeX) {
    // Note that sizeX is the size of a Row, not the number of rows
    return tensor[x * (sizeX)+ y];
}

inline void twoDimWrite(std::vector<float> &tensor, int &x, int &y, const int &sizeX, float &val) {
    tensor[x * (sizeX) + y] = val;
}

// Step #2: Implement Read/Write Accessors for a 4D Tensor
inline float fourDimRead(std::vector<float> &tensor, int &x, int &y, int &z, int &b, 
        const int &sizeX, const int &sizeY, const int &sizeZ) {
    return tensor[(x * sizeX * sizeY * sizeZ) + (y * sizeY * sizeZ) + (z * sizeZ) + b];
}

inline void fourDimWrite(std::vector<float> &tensor, int &x, int &y, int &z, int &b, 
        const int &sizeX, const int &sizeY, const int &sizeZ, float &val) {
    tensor[(x * sizeX * sizeY * sizeZ) + (y * sizeY * sizeZ) + (z * sizeZ) + b] = val;
}

// DO NOT EDIT THIS FUNCTION //
std::vector<float> formatTensor(torch::Tensor tensor) {
    tensor = tensor.flatten();
    tensor = tensor.contiguous();
    std::vector<float> vec(tensor.data_ptr<float>(), tensor.data_ptr<float>() + tensor.numel());
    return vec;
}

/* Programming Your Attention Modules.
 * 
 * You are given Q, K, and V Tensors as inputs that are formatted as vectors. We have also created O and QK^t Tensors 
 * that are formatted as vectors. After you have implemented your accessors in the Warm-Up you should be able to
 * read/write to these tensors via the read/write functions above.
 *
 * You are also given 4 integers as parameters: B, H, N, d:
 *
 * B (Batch Size) - The number of samples for your attention layer. Think of it this way - if I asked my dnn
 * a question and it output 5 different answers it had a batch size of 5. These samples are independent of each
 * other and thus can be parallelized.
 *
 * H (Number of Heads) - Each head runs on its own set of Q, K, V matrices. This effectively allows each head
 * to operate the same attention algorithm, but each with each head using different hyperparameters. These
 * allow each head to have their own definition of what relevance is when looking at a token. These heads
 * can operate independently of one another and thus can be parallized.
 *
 * N (Sequence Length) - The number of tokens. You may think of this as the number of words in a sample.
 *
 * d (Embedding Dimensionality) - The number of features each token encodes per attention head. Let's
 * say I encoded a word using the follow (length, number of vowels, has a capital letters). The
 * emvedded dimensionaliy would be 3.
 * */

// ---------------------------------------------------------- //
//                  PART 1: NAIVE ATTENTION                   //
// ---------------------------------------------------------- //

torch::Tensor myNaiveAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor QK_tTensor,
                int B, int H, int N, int d){

    // Q, K, V are passed in with Shape: (B, H, N, d)
    //QK^t Intermediate Tensor has Shape (N, N)
    
    //Make O Tensor with Shape (B, H, N, d) 
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    //Format O, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    //Format QK_t Tensor into a 2D vector.
    std::vector<float> QK_t = formatTensor(QK_tTensor);

    // (B, H, N, d) @ (B, H, d, N) --> (B, H, N, N)

    for (int batch = 0; batch < B; batch++) {
        for (int head = 0; head < H; head++) {
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    float out = 0;
                    for (int k = 0; k < d; k++) {
                        float q_val = fourDimRead(Q, batch, head, i, k, H, N, d);
                        float k_val = fourDimRead(K, batch, head, j, k, H, N, d);
                        out += q_val * k_val;
                    }
                    twoDimWrite(QK_t, i, j, N, out);
                }
            }

            for (int row = 0; row < N; row++) {
                float tot = 0;
                for (int col = 0; col < N; col++) {
                    float exp_val = exp(twoDimRead(QK_t, row, col, N));
                    tot += exp_val;
                    twoDimWrite(QK_t, row, col, N, exp_val);
                }

                for (int col = 0; col < N; col++) {
                    float exp_val = twoDimRead(QK_t, row, col, N) / tot;
                    twoDimWrite(QK_t, row, col, N, exp_val);
                }

            }

            // (N, N) @ (N, d) --> (N, d)
            for (int row = 0; row < N; row++) {
                for (int col = 0; col < d; col++) { 
                    float out = 0;
                    for (int sel = 0; sel < N; sel++) {
                        float a_val = twoDimRead(QK_t, row, sel, N);
                        float v_val = fourDimRead(V, batch, head, sel, col, H, N, d);
                        out += a_val * v_val;
                    }
                    fourDimWrite(O, batch, head, row, col, H, N, d, out);
                }
            }
        }
    }
   
    /* Here is an example of how to read/write 0's to  Q (B, H, N, d) using the 4D accessors

        //loop over Batch Size
         for (int b = 0; b < B; b++) {

             //loop over Heads
             for (int h = 0; h < H; h++) {

                 //loop over Sequence Length
                 for (int i = 0; i < N; i++) {

                     //loop over Embedding Dimensionality
                     for (int j = 0; j < d; j++) {
                        float val = fourDimRead(Q, b, h, i, j, H, N, d);
                        val = 0.0;
                        fourDimWrite(Q, b, h, i, j, H, N, d, val);
                     }
                 }
             }
         }
    */

    /* Here is an example of how to read/write 0's to  QK_t (N, N) using the 2D accessors

           for (int i = 0; i < N; i++) {
	       for (int j = 0; j < N; j++) {
	           float val = twoDimRead(QK_t, i, j, N);
               val = 0.0;
	           twoDimWrite(QK_t, i, j, N, val);
             }
         }
    */
    
    // -------- YOUR CODE HERE  -------- //
    
    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//     PART 2: BLOCKED MATRIX MULTIPLY AND UNFUSED SOFTMAX    //
// ---------------------------------------------------------- //

torch::Tensor myUnfusedAttentionBlocked(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor QK_tTensor,
                int B, int H, int N, int d){
    
    // Q, K, V are passed in with Shape: (B, H, N, d)
    //QK^t Intermediate Tensor has Shape (N, N)

    //Make O Tensor with Shape (B, H, N, d) 
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    //Format O, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    //Format QK_t Tensor into a 2D vector.
    std::vector<float> QK_t = formatTensor(QK_tTensor);

    // -------- YOUR CODE HERE  -------- //
    // NOTE: what are the right units here? I think bytes
    int cache_line_size = 64;
    int block_size = cache_line_size / sizeof(float); 

    // printf("B: %d H: %d N: %d d: %d\n", B, H, N, d);

    for (int batch = 0; batch < B; batch++) {
        for (int head = 0; head < H; head++) {
            // compute QK_t

            std::fill(QK_t.begin(), QK_t.end(), 0.0);
            // (N, d) @ (d, N) --> (N, N) N.B we are operating on K in col-major manner
            for (int row_block = 0; row_block < N; row_block+=block_size) {
                for (int col_block = 0; col_block < N; col_block += block_size) {
                    for (int sel_block = 0; sel_block < d; sel_block += block_size) {
                        for (int row = row_block; row < std::min(N, row_block + block_size); row++) {
                            for (int col = col_block; col < std::min(N, col_block + block_size); col++) {
                                float out = twoDimRead(QK_t, row, col, N); // NOTE: is QK_t initialised to zero?

                                for (int sel = sel_block; sel < std::min(d, sel_block + block_size); sel++) {
                                    // printf("attention on batch: %d head: %d row: %d col: %d sel: %d \n", batch, head, row, col, sel);
                                    float q_val = fourDimRead(Q, batch, head, row, sel, H, N, d);
                                    float k_val = fourDimRead(K, batch, head, col, sel, H, N, d);
                                    out += q_val * k_val;
                                }

                                twoDimWrite(QK_t, row, col, N, out);
                            }
                        }
                    }
                }
            }

            // compute softmax
            for (int row = 0; row < N; row++) {
                float tot = 0;
                for (int col = 0; col < N; col++) {
                    float exp_val = exp(twoDimRead(QK_t, row, col, N));
                    tot += exp_val;
                    twoDimWrite(QK_t, row, col, N, exp_val);
                }

                for (int col = 0; col < N; col++) {
                    float exp_val = twoDimRead(QK_t, row, col, N) / tot;
                    twoDimWrite(QK_t, row, col, N, exp_val);
                }
            }

            // compute O

            // (N, N) @ (N, d) -> (N, d)
            // NOTE: would it be better to compute the transpose of QK_t above so we can access it col-major?
            for (int row_block = 0; row_block < N; row_block += block_size) {
                for (int col_block = 0; col_block < d; col_block += block_size) {
                    for (int sel_block = 0; sel_block < N; sel_block += block_size) {
                        for (int row = row_block; row < std::min(N, row_block + block_size); row++) {
                            for (int col = col_block; col < std::min(d, col_block + block_size); col++) {

                                float out = fourDimRead(O, batch, head, row, col, H, N, d);
                                for (int sel = sel_block; sel < std::min(N, sel_block + block_size); sel++) {
                                    float atten_val = twoDimRead(QK_t, row, sel, N); 
                                    float v_val = fourDimRead(V, batch, head, sel, col, H, N, d); // I think this must be a bit wrong because we're fucking the cache lines
                                    out += atten_val * v_val;
                                }
                                fourDimWrite(O, batch, head, row, col, H, N, d, out);
                            }
                        }
                    }
                }
            }

        }
    }

    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//                 PART 3: FUSED ATTENTION     	              //
// ---------------------------------------------------------- //

torch::Tensor myFusedAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor temp,
                int B, int H, int N, int d){

    // Q, K, V are passed in with Shape: (B, H, N, d)

    //Make O Tensor with Shape (B, H, N, d)
    //and O Row Tensor with Shape (N)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);
    at::Tensor ORowTensor = at::zeros({N}, at::kFloat);

    //Format Y, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);
    
    //Format ORow Tensor into a 1D vector
    // You can simply access this as ORow[i]
    std::vector<float> ORow = formatTensor(ORowTensor);


    // -------- YOUR CODE HERE  -------- //
    // We give you a template of the first three loops for your convenience
    //loop over batch
    #pragma omp parallel for collapse(3)
    for (int batch = 0; batch < B; batch++){

        //loop over heads
        for (int head = 0; head < H; head++){
            for (int row = 0; row < N ; row++){

		// YRow is moved inside so each OpenMP thread gets a local copy.
                at::Tensor ORowTensor = temp.index({torch::indexing::Slice(omp_get_thread_num(), torch::indexing::None)});      
                std::vector<float> ORow = formatTensor(ORowTensor);
		//YOUR CODE HERE
                float row_sum = 0;
                for (int col = 0; col < N; col++) {
                    float out = 0;
                    for (int sel = 0; sel < d; sel++) {
                        float q_val = fourDimRead(Q, batch, head, row, sel, H, N, d);
                        float k_val = fourDimRead(K, batch, head, col, sel, H, N, d);
                        out += q_val * k_val;
                    }
                    out = exp(out);
                    row_sum += out;
                    ORow[col] = out; 
                }

                for (int col = 0; col < d; col++) {
                    float out = 0;
                    for (int sel = 0; sel < N; sel++) {
                        float v_val = fourDimRead(V, batch, head, sel, col, H, N, d);
                        out += (ORow[sel]/row_sum) * v_val;
                    }
                    fourDimWrite(O, batch, head, row, col, H, N, d, out);
                }
            }
	    }
    }
	    
	
    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//                PART 4: FLASH ATTENTION 		      //
// ---------------------------------------------------------- //


torch::Tensor myFlashAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor,
               torch::Tensor QiTensor, torch::Tensor KjTensor, torch::Tensor VjTensor,
               torch::Tensor SijTensor, torch::Tensor PijTensor, torch::Tensor PVTensor,
               torch::Tensor OiTensor, torch::Tensor LTensor,  torch::Tensor LiTensor, 
	       torch::Tensor LijTensor, torch::Tensor LnewTensor, int Bc, int Br,
                int B, int H, int N, int d) {
        
    // Q, K, V are passed in with Shape: (B, H, N, d)
    // Sij, Pij are passed in with Shape: (Br, Bc)
    // Kj, Vj are passed in with Shape: (Bc, d)
    // Qi, Oi, and PV  are passed in with Shape: (Br, d)
    // L in passed in with Shape: (N)
    // Li, Lij, and Lnew are passed in with shape (Br)

    //Make O Tensor with Shape (B, H, N, d)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);
   
    //Format All Tensors into Vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);
    std::vector<float> Sij = formatTensor(SijTensor);
    std::vector<float> Pij = formatTensor(PijTensor);
    std::vector<float> Kj = formatTensor(KjTensor);
    std::vector<float> Vj = formatTensor(VjTensor);
    std::vector<float> Qi = formatTensor(QiTensor);
    std::vector<float> Oi = formatTensor(OiTensor);
    std::vector<float> l = formatTensor(LTensor);
    std::vector<float> PV = formatTensor(PVTensor);
    std::vector<float> li = formatTensor(LiTensor);
    std::vector<float> lij = formatTensor(LijTensor);
    std::vector<float> lnew = formatTensor(LnewTensor);

    int t_idx = 5;

    // -------- YOUR CODE HERE  -------- //
    printf("B: %d H: %d N: %d d: %d Br: %d Bc: %d\n", B, H, N, d, Br, Bc);
    for (int batch = 0; batch < B; batch++) {
        for (int head = 0; head < H; head++) {

            for (int col_block = 0; col_block < N; col_block+=Bc) {
                for (int row_block = 0; row_block < N; row_block += Br) {

                    for (int col = col_block; col < std::min(col_block + Bc, N); col++) {
                        for (int row = row_block; row < std::min(row_block + Br, N); row++) {
                            float out = 0;
                            // compute QKt
                            for (int sel = 0; sel < d; sel++) {
                                float q_val = fourDimRead(Q, batch, head, row, sel, H, N, d);
                                float k_val = fourDimRead(K, batch, head, col, sel, H, N, d);
                                out += q_val * k_val;
                            }
                            float exp_out = exp(out);
                            int r_idx = row - row_block;
                            int c_idx = col - col_block;
                            twoDimWrite(Pij, r_idx, c_idx, Bc, exp_out);
                        }
                    }


                    for (int r_idx = 0; r_idx < std::min(Br, N - row_block); r_idx++) {
                        float row_sum = 0;
                        for (int c_idx = 0; c_idx < std::min(Bc, N - col_block); c_idx++) {
                            row_sum += twoDimRead(Pij, r_idx, c_idx, Bc);
                        }

                        int row = r_idx + row_block;
                        float l_old = l[row];
                        float l_new = l_old + row_sum;

                        for (int c_idx = 0; c_idx < d; c_idx++) {
                            float out = 0;
                            for (int sel = 0; sel < std::min(Bc, N - col_block); sel++) {
                                float pij_val = twoDimRead(Pij, r_idx, sel, Bc);
                                int row = sel + col_block;
                                float v_val = fourDimRead(V, batch, head, row, c_idx, H, N, d); 
                                out += pij_val * v_val;
                            }
                            float o_val = fourDimRead(O, batch, head, row, c_idx, H, N, d);
                            o_val *= l_old;
                            o_val += out;
                            o_val /= l_new;
                            fourDimWrite(O, batch, head, row, c_idx, H, N, d, o_val);
                            l[row] = l_new;
                        }
                    }

                }
            }

            std::fill(l.begin(), l.end(), 0);
        }
    }

    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}

/* DO NOT EDIT THESE BINDINGS */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("myNaiveAttention", &myNaiveAttention, "Naive Attention");
  m.def("myUnfusedAttentionBlocked", &myUnfusedAttentionBlocked, " Blocked Unfused Attention");
  m.def("myFusedAttention", &myFusedAttention, "Fused Attention");
  m.def("myFlashAttention", &myFlashAttention, "Flash Attention");
  m.def("twoDimRead", &twoDimRead, "twoDimRead");
  m.def("fourDimRead", &fourDimRead, "fourDimRead");
}
