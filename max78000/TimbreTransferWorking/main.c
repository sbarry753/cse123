/******************************************************************************
 *
 * Copyright (C) 2022-2023 Maxim Integrated Products, Inc. (now owned by 
 * Analog Devices, Inc.),
 * Copyright (C) 2023-2024 Analog Devices, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 ******************************************************************************/

/**
 * @file        main.c
 * @brief       FreeRTOS Example Application.
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "FreeRTOS.h"
#include "FreeRTOSConfig.h"
#include "portmacro.h"
#include "task.h"
#include "semphr.h"
#include "FreeRTOS_CLI.h"
#include "mxc_device.h"
#include "wut.h"
#include "uart.h"
#include "lp.h"
#include "led.h"
#include "board.h"
#include "mxc.h"
#include "max9867.h"
#include "buffer.h"

/* FreeRTOS+CLI */
//void vRegisterCLICommands(void);

/* Mutual exclusion (mutex) semaphores */
SemaphoreHandle_t xGPIOmutex;

/* Task IDs */
TaskHandle_t cmd_task_id;

/* Enables/disables tick-less mode */
unsigned int disable_tickless = 1;

/* Stringification macros */
#define STRING(x) STRING_(x)
#define STRING_(x) #x

/* Console ISR selection */
#if (CONSOLE_UART == 0)
#define UARTx_IRQHandler UART0_IRQHandler
#define UARTx_IRQn UART0_IRQn
mxc_gpio_cfg_t uart_cts = { MXC_GPIO0, MXC_GPIO_PIN_2, MXC_GPIO_FUNC_IN, MXC_GPIO_PAD_WEAK_PULL_UP,
                            MXC_GPIO_VSSEL_VDDIOH };
mxc_gpio_cfg_t uart_rts = { MXC_GPIO0, MXC_GPIO_PIN_3, MXC_GPIO_FUNC_OUT, MXC_GPIO_PAD_NONE,
                            MXC_GPIO_VSSEL_VDDIOH };
#else
#error "Please update ISR macro for UART CONSOLE_UART"
#endif
mxc_uart_regs_t *ConsoleUART = MXC_UART_GET_UART(CONSOLE_UART);

mxc_gpio_cfg_t uart_cts_isr;

/* Array sizes */
#define CMD_LINE_BUF_SIZE 80
#define OUTPUT_BUF_SIZE 512

void passthrough_task(void* pvParameters);

/* Defined in freertos_tickless.c */
extern void wutHitSnooze(void);

// ADDED: Circular buffer test configuration
#define BUFFER_SIZE 10
#define PRODUCERS 2
#define CONSUMERS 3
#define MAX_PUTS_PER_PRODUCER 10
const char charset[] = "abcdefghijklmnopqrstuvwxyz";
Buffer testBuffer;

// ADDED: Helper function for buffer test
char* random_string(void) 
{
  int length = (rand() % 6) + 3;
  char *str = pvPortMalloc(length + 1);
  if (!str) return NULL;
  
  for (int i = 0; i < length; i++) {
    str[i] = charset[rand() % (sizeof(charset) - 1)];
  }
  str[length] = '\0';
  return str;
}

// ADDED: Producer task
void producer_task(void *arg)
{
  long tid = (long) arg;
  char* str;
  printf("[BUFFER TEST] Producer %ld started\n", tid);

  for (int i = 0; i < MAX_PUTS_PER_PRODUCER; i++) {
    str = random_string();
    if (str) {
      buffer_put(&testBuffer, str);
      printf("[BUFFER TEST] => P%ld: %s\n", tid, str);
    }
    vTaskDelay(pdMS_TO_TICKS(100));
  }

  printf("[BUFFER TEST] Producer %ld finished\n", tid);
  vTaskDelete(NULL);
}

// ADDED: Consumer task
void consumer_task(void *arg)
{
  long tid = (long) arg;
  char* str;
  printf("[BUFFER TEST] Consumer %ld started\n", tid);

  for (int i = 0; i < (PRODUCERS * MAX_PUTS_PER_PRODUCER) / CONSUMERS; i++) {
    str = buffer_get(&testBuffer);
    printf("[BUFFER TEST] <= C%ld: %s\n", tid, str);
    vPortFree(str);
    vTaskDelay(pdMS_TO_TICKS(200));
  }

  printf("[BUFFER TEST] Consumer %ld finished\n", tid);
  vTaskDelete(NULL);
}

/***** Functions *****/

/* =| UART0_IRQHandler |======================================
 *
 * This function overrides the weakly-declared interrupt handler
 *  in system_max326xx.c and is needed for asynchronous UART
 *  calls to work properly
 *
 * ===========================================================
 */
void UARTx_IRQHandler(void)
{
    MXC_UART_AsyncHandler(ConsoleUART);
    wutHitSnooze();
}


#if configUSE_TICKLESS_IDLE
/* =| freertos_permit_tickless |==========================
 *
 * Determine if any hardware activity should prevent
 *  low-power tickless operation.
 *
 * =======================================================
 */
int freertos_permit_tickless(void)
{
    if (disable_tickless == 1) {
        return E_BUSY;
    }

    return MXC_UART_GetActive(ConsoleUART);
}
#endif

/* =| WUT_IRQHandler |==========================
 *
 * Interrupt handler for the wake up timer.
 *
 * =======================================================
 */
void WUT_IRQHandler(void)
{
    MXC_WUT_ClearFlags();
    NVIC_ClearPendingIRQ(WUT_IRQn);
}

/* =| main |==============================================
 *
 * This program demonstrates FreeRTOS tasks, mutexes,
 *  and the FreeRTOS+CLI extension.
 *
 * =======================================================
 */
int main(void)
{
    /* Delay to prevent bricks */
    MXC_Delay(MXC_DELAY_MSEC(200));

    /* Setup manual CTS/RTS to lockout console and wake from deep sleep */
    MXC_GPIO_Config(&uart_cts);
    MXC_GPIO_Config(&uart_rts);

    /* Enable incoming characters */
    MXC_GPIO_OutClr(uart_rts.port, uart_rts.mask);

    /* Print banner (RTOS scheduler not running) */
    printf("\n-=- %s FreeRTOS (%s) Demo -=-\n", STRING(TARGET), tskKERNEL_VERSION_NUMBER);
#if configUSE_TICKLESS_IDLE
    printf("Tickless idle is configured. Type 'tickless 1' to enable.\n");
#endif
    printf("SystemCoreClock = %d\n", SystemCoreClock);

    // ADDED: Initialize buffer and random seed
    printf("[BUFFER TEST] Initializing circular buffer (size=%d)\n", BUFFER_SIZE);
    buffer_init(&testBuffer, BUFFER_SIZE);
    srand(xTaskGetTickCount());

    /* Create mutexes */
    xGPIOmutex = xSemaphoreCreateMutex();
    if (xGPIOmutex == NULL) {
        printf("xSemaphoreCreateMutex failed to create a mutex.\n");
    } else {
	if (xTaskCreate(passthrough_task, "passthrough", configMINIMAL_STACK_SIZE * 4, NULL,
				tskIDLE_PRIORITY + 2, NULL) != pdPASS) {
		printf("Failed to create passthrough task. \n");
	}

        // ADDED: Create buffer test tasks ONLY
        printf("[BUFFER TEST] Creating %d producers and %d consumers\n", PRODUCERS, CONSUMERS);
        for (long i = 0; i < PRODUCERS; i++) {
            if (xTaskCreate(producer_task, "producer", configMINIMAL_STACK_SIZE * 2, 
                           (void*)i, tskIDLE_PRIORITY + 1, NULL) != pdPASS) {
                printf("[BUFFER TEST] Failed to create producer %ld\n", i);
            }
        }
        for (long i = 0; i < CONSUMERS; i++) {
            if (xTaskCreate(consumer_task, "consumer", configMINIMAL_STACK_SIZE * 2, 
                           (void*)i, tskIDLE_PRIORITY + 1, NULL) != pdPASS) {
                printf("[BUFFER TEST] Failed to create consumer %ld\n", i);
            }
        }

        /* Start scheduler */
        printf("Starting scheduler.\n");
        vTaskStartScheduler();
    }

    /* This code is only reached if the scheduler failed to start */
    printf("ERROR: FreeRTOS did not start due to above error!\n");
    while (1) {
        __NOP();
    }

    /* Quiet GCC warnings */
    return -1;
}

// --------------- Audio passthrough program ---------------------- //

#undef USE_I2S_INTERRUPTS

#define CODEC_I2C MXC_I2C1
#define CODEC_I2C_FREQ 100000

#define CODEC_MCLOCK 12288000
#define SAMPLE_RATE 48000

#define I2S_DMA_BUFFER_SIZE 64

volatile int dma_flag;
uint32_t i2s_rx_buffer[I2S_DMA_BUFFER_SIZE * 2];
int dma_ch_tx, dma_ch_rx;
uint32_t *rxBufPtr = i2s_rx_buffer;

void blink_halt(const char *msg)
{
    if (msg && *msg)
        puts(msg);

    for (;;) {
        LED_On(LED1);
        LED_Off(LED2);
        MXC_Delay(MXC_DELAY_MSEC(500));
        LED_On(LED2);
        LED_Off(LED1);
        MXC_Delay(MXC_DELAY_MSEC(500));
    }
}

void dma_handler(void)
{
    dma_flag = 1;
    MXC_DMA_Handler();
}

void dma_init(void)
{
    MXC_NVIC_SetVector(DMA0_IRQn, dma_handler);
    MXC_NVIC_SetVector(DMA1_IRQn, dma_handler);
    NVIC_EnableIRQ(DMA0_IRQn);
    NVIC_EnableIRQ(DMA1_IRQn);
}

void dma_callback(int channel, int result)
{
    static uint32_t *tx_buf = i2s_rx_buffer + I2S_DMA_BUFFER_SIZE;

    if (channel == dma_ch_tx) {
        MXC_DMA_ReleaseChannel(dma_ch_tx);
        dma_ch_tx = MXC_I2S_TXDMAConfig(tx_buf, I2S_DMA_BUFFER_SIZE * sizeof(i2s_rx_buffer[0]));

    } else if (channel == dma_ch_rx) {
        tx_buf = rxBufPtr;

        if (rxBufPtr == i2s_rx_buffer) {
            rxBufPtr = i2s_rx_buffer + I2S_DMA_BUFFER_SIZE;
        } else {
            rxBufPtr = i2s_rx_buffer;
        }
        MXC_DMA_ReleaseChannel(dma_ch_rx);
        dma_ch_rx = MXC_I2S_RXDMAConfig(rxBufPtr, I2S_DMA_BUFFER_SIZE * sizeof(i2s_rx_buffer[0]));
    }
}

void dma_work_loop(void)
{
    int trig = 0;

    dma_init();
    MXC_I2S_RegisterDMACallback(dma_callback);
    dma_ch_tx = MXC_I2S_TXDMAConfig(i2s_rx_buffer + I2S_DMA_BUFFER_SIZE,
                                    I2S_DMA_BUFFER_SIZE * sizeof(i2s_rx_buffer[0]));
    dma_ch_rx = MXC_I2S_RXDMAConfig(i2s_rx_buffer, I2S_DMA_BUFFER_SIZE * sizeof(i2s_rx_buffer[0]));

    for (;;) {
        if (dma_flag) {
            dma_flag = 0;
            /*
        dma activity triggered work
      */
            if (++trig == SAMPLE_RATE / I2S_DMA_BUFFER_SIZE) {
                trig = 0;
                LED_Toggle(LED2);
            }
        }
        /*
      non-dma activity triggered work
    */
    }
}

void i2c_init(void)
{
    if (MXC_I2C_Init(CODEC_I2C, 1, 0) != E_NO_ERROR)
        blink_halt("Error initializing I2C controller");
    else
        printf("I2C initialized successfully \n");

    MXC_I2C_SetFrequency(CODEC_I2C, CODEC_I2C_FREQ);
}

void codec_init(void)
{
    if (max9867_init(CODEC_I2C, CODEC_MCLOCK, 1) != E_NO_ERROR)
        blink_halt("Error initializing MAX9867 CODEC");


    if (max9867_enable_playback(1) != E_NO_ERROR)
        blink_halt("Error enabling playback path");

    printf("max9867_enable_playback() successful exit\n");


    if (max9867_playback_volume(-6, -6) != E_NO_ERROR)
        blink_halt("Error setting playback volume");

    printf("max9867_playback_volume() successful exit\n");


    if (max9867_enable_record(1) != E_NO_ERROR)
        blink_halt("Error enabling record path");

    printf("max9867_enable_record() successful exit\n");

    if (max9867_adc_level(-12, -12) != E_NO_ERROR)
        blink_halt("Error setting ADC level");

    printf("max9867_adc_level() successful exit\n");

    if (max9867_linein_gain(-6, -6) != E_NO_ERROR)
        blink_halt("Error setting Line-In gain");
    else {
        printf("max9867_linein_gain() successful exit\n");
        printf("Codec initialized successfully \n");
    }
}

void i2s_init(void)
{
    mxc_i2s_req_t req;

#define I2S_CRUFT_PTR (void *)UINT32_MAX
#define I2S_CRUFT_LEN UINT32_MAX

    req.wordSize = MXC_I2S_WSIZE_WORD;
    req.sampleSize = MXC_I2S_SAMPLESIZE_TWENTYFOUR;
    req.bitsWord = 24;
    req.adjust = MXC_I2S_ADJUST_LEFT;
    req.justify = MXC_I2S_MSB_JUSTIFY;
    req.wsPolarity = MXC_I2S_POL_NORMAL;
    /* I2S Peripheral is in slave mode - no need to set clkdiv */
    req.channelMode = MXC_I2S_EXTERNAL_SCK_EXTERNAL_WS;
    req.stereoMode = MXC_I2S_STEREO;

    req.bitOrder = MXC_I2S_MSB_FIRST;

    req.rawData = NULL;
    req.txData = I2S_CRUFT_PTR;
    req.rxData = I2S_CRUFT_PTR;
    req.length = I2S_CRUFT_LEN;

    if (MXC_I2S_Init(&req) != E_NO_ERROR)
        blink_halt("Error initializing I2S");
    else
        printf("I2S initialized successfully \n");
}

void passthrough_task(void* pvParameters)
{
#if defined(BOARD_FTHR_REVA)
    /* Wait for PMIC 1.8V to become available, about 180ms after power up. */
    MXC_Delay(MXC_DELAY_MSEC(200));
#endif

    /* Switch to 100 MHz clock */
    MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO);
    SystemCoreClockUpdate();

    printf("***** MAX9867 CODEC DMA Loopback Example *****\n");

    printf("Waiting...\n");

    /* DO NOT DELETE THIS LINE: */
    MXC_Delay(MXC_DELAY_SEC(2)); /* Let debugger interrupt if needed */

    printf("Running...\n");

    i2c_init();
    codec_init();
    i2s_init();

    dma_work_loop();
}
