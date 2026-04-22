#include "arm_math.h"

/*
  Compatibility shim:

  Your CMSIS-DSP Source tree does NOT include arm_cfft_init_f32.c,
  but arm_rfft_fast_init_f32.c calls arm_cfft_init_f32().

  Older CMSIS-DSP versions use prebuilt const structs (arm_cfft_sR_f32_lenXXX).
  This shim selects the correct one at runtime.
*/

arm_status arm_cfft_init_f32(arm_cfft_instance_f32 *S, uint16_t fftLen)
{
    switch(fftLen)
    {
        case 16:   *S = arm_cfft_sR_f32_len16;   break;
        case 32:   *S = arm_cfft_sR_f32_len32;   break;
        case 64:   *S = arm_cfft_sR_f32_len64;   break;
        case 128:  *S = arm_cfft_sR_f32_len128;  break;
        case 256:  *S = arm_cfft_sR_f32_len256;  break;
        case 512:  *S = arm_cfft_sR_f32_len512;  break;
        case 1024: *S = arm_cfft_sR_f32_len1024; break;
        case 2048: *S = arm_cfft_sR_f32_len2048; break;
        case 4096: *S = arm_cfft_sR_f32_len4096; break;
        default:   return ARM_MATH_ARGUMENT_ERROR;
    }
    return ARM_MATH_SUCCESS;
}