/*
 * main.c
 * Testing circular buffer that I converted to FreeRTOS
 * Random length random strings produced/consumed via circular buffer
 * FreeRTOS version
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "FreeRTOS.h"
#include "task.h"
#include "buffer.h"
#include "mxc.h"

#define PRODUCERS    2
#define CONSUMERS    5

#define MAX_SLEEP_MS 600
#define MAX_PUTS_PER_PRODUCER  24

#define BUFFER_SIZE 10

const char charset[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

Buffer buffer;

char* random_string() 
{
  int length = (rand() % 11) + 5; 
  char *str = pvPortMalloc(length + 1);
  if (!str) return NULL;
  
  for (int i = 0; i < length; i++) {
    str[i] = charset[rand() % (sizeof(charset) - 1)];
  }
  str[length] = '\0';
  return str;
}

static void produce_task(void *arg)
{
  long tid = (long) arg;
  char* str;
  printf("P %ld running\n", tid);

  for (int i = 0; i < MAX_PUTS_PER_PRODUCER; i++) {
    str = random_string();
    if (str) {
      buffer_put(&buffer, str);
      printf("=> P %ld produced %s\n", tid, str);
    }
  }

  printf("P %ld ### finished ###\n", tid);
  vTaskDelete(NULL); // Task ends itself
}

static void consume_task(void *arg)
{
  long tid = (long) arg;
  int millis;
  char* str;
  printf("C %ld running\n", tid);

  for (;;) {
    millis = rand() % MAX_SLEEP_MS;
    str = buffer_get(&buffer);
    printf("<= C %ld consumed %s - sleeping for %dms\n", tid, str, millis);
    vPortFree(str);
    vTaskDelay(pdMS_TO_TICKS(millis)); // FreeRTOS delay
  }

  // Will never return, consumers are in an infinite loop
}

int main(void)
{
  /* Initialize hardware */
  MXC_Delay(MXC_DELAY_MSEC(200));
  MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO);
  SystemCoreClockUpdate();

  printf("***** Circular Buffer FreeRTOS Demo *****\n");

  /* Initialize buffer */
  buffer_init(&buffer, BUFFER_SIZE);

  /* Seed random */
  srand(xTaskGetTickCount());

  /* Create consumer tasks */
  for (long tid = 0; tid < CONSUMERS; tid++) {
    char name[16];
    snprintf(name, sizeof(name), "Consumer%ld", tid);
    xTaskCreate(consume_task, name, configMINIMAL_STACK_SIZE, 
                (void*)tid, tskIDLE_PRIORITY + 1, NULL);
  }

  /* Create producer tasks */
  for (long tid = 0; tid < PRODUCERS; tid++) {
    char name[16];
    snprintf(name, sizeof(name), "Producer%ld", tid);
    xTaskCreate(produce_task, name, configMINIMAL_STACK_SIZE, 
                (void*)tid, tskIDLE_PRIORITY + 1, NULL);
  }

  /* Start FreeRTOS scheduler */
  printf("Starting scheduler\n");
  vTaskStartScheduler();

  /* Should never reach here */
  printf("ERROR: FreeRTOS scheduler failed to start!\n");
  while (1) {}

  return 0;
}
