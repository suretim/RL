#pragma once

#include <stdint.h>
#include <stddef.h>
#include "esp_err.h"
#include "version.h"
enum CaseType { PPO_CASE,NN_PPO_CASE, META_CASE ,IMG_CASE,NUM_CASE};


//#define PPO_CASE 0
//#define META_CASE 1
//#define IMG_CASE 2
//#define NN_PPO_CASE 3
//#define INIT_CASE 4

#define INFER_CASE PPO_CASE

#define MAX_FEATURE_DIM 64
#define MAX_NUM_CLASSES  PORT_CNT
#define MAX_SEQ_LEN 64

// 全局参数 (需在你的工程中定义实际大小)
 
#define PPO_FEATURE_DIM STATE_CNT 
#define NN_PPO_FEATURE_DIM STATE_CNT
#define META_FEATURE_DIM STATE_CNT
#define IMG_FEATURE_DIM 64 
 
#define PPO_CLASSES PORT_CNT
#define NN_PPO_CLASSES PORT_CNT
#define META_CLASSES 3
#define IMG_CLASSES 3

#define PPO_SEQ_LEN 1 
#define NN_PPO_SEQ_LEN 10 
#define META_SEQ_LEN 10
#define IMG_SEQ_LEN 64

 #ifndef INFER_CASE
  #error "INFER_CASE is not defined"
#elif INFER_CASE == PPO_CASE
  #define FEATURE_DIM (PPO_FEATURE_DIM)
  #define NUM_CLASSES (PPO_CLASSES)
  #define SEQ_LEN (PPO_SEQ_LEN)
#elif INFER_CASE == NN_PPO_CASE
  #define FEATURE_DIM (NN_PPO_FEATURE_DIM)
  #define NUM_CLASSES (NN_PPO_CLASSES)
  #define SEQ_LEN (NN_PPO_SEQ_LEN)
#elif INFER_CASE == META_CASE
  #define FEATURE_DIM (META_FEATURE_DIM)
  #define NUM_CLASSES (META_CLASSES)
  #define SEQ_LEN (META_SEQ_LEN)
#elif INFER_CASE == IMG_CASE
  #define FEATURE_DIM (IMG_FEATURE_DIM)
  #define NUM_CLASSES (IMG_CLASSES)
  #define SEQ_LEN (IMG_SEQ_LEN)
#else
  #error "Unknown INFER_CASE"
#endif

 
 struct CLASSIFIER_Prams {
    int infer_case; // 0: PPO, 1: META
    int seq_len;
    int feature_dim;
    int num_classes;
 };

 

// 编译期校验（C11 _Static_assert 或 C++ static_assert）
_Static_assert(FEATURE_DIM > 0 && NUM_CLASSES > 0, "dims must be positive");
_Static_assert(sizeof(float) == 4, "float must be 32-bit");

 
#define FISHER_LAYER 12
//void classifier_set_params(const float *weights, const float *bias, int input_dim, int output_dim);
int classifier_predict(const float *features);  // 返回预测类别索引
void update_classifier_weights_bias(const float* values, int value_count,int type) ;
void update_fishermatrix_theta(const float* values, int value_count,int type) ;

/**
 * @brief 从 buffer 更新 classifier (仅内存，不写入 NVS)
 */
void set_classifier_from_buffer(const uint8_t* buf, size_t len,size_t type );

/**
 * @brief 从二进制 buffer 更新 classifier 并写入 NVS
 * 
 * @param data  输入的二进制数据
 * @param len   数据长度 (字节)
 * @return int  0 表示成功，-1 表示失败
 */
int update_classifier_from_bin(const uint8_t* data, size_t len,size_t type);

/**
 * @brief 从 NVS 恢复 classifier 参数 (仅内存，不写回 NVS)
 */
//int restore_classifier_from_nvs(size_t type);

  int safe_nvs_operation(size_t type ) ;

#ifdef __cplusplus
extern "C" {
#endif
   void   initialize_nvs_robust(void);
  
void init_classifier_from_header(void) ;
 
#ifdef __cplusplus
}
#endif

