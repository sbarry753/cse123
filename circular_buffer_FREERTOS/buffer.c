#include <stdlib.h>
#include <stdio.h>
#include "buffer.h"
#include "FreeRTOS.h"
#include "semphr.h"

/*
 * Initialize the circular bounded buffer B 
 */
void buffer_init(Buffer* b, size_t capacity)
{
  b->capacity = capacity;
  b->next_in = 0;
  b->next_out = 0;
  
  /* Allocate buffer using FreeRTOS heap */
  b->buffer = (void**) pvPortMalloc(b->capacity * sizeof(void*));
  if (!b->buffer) {
    /* In FreeRTOS, can't use fprintf/exit - handle error appropriately */
    configASSERT(0); /* Triggers assertion failure */
    return;
  }
  
  /* Create counting semaphore for occupied slots (initially 0) */
  b->occupied = xSemaphoreCreateCounting(capacity, 0);
  configASSERT(b->occupied != NULL);
  
  /* Create counting semaphore for vacant slots (initially full capacity) */
  b->vacant = xSemaphoreCreateCounting(capacity, capacity);
  configASSERT(b->vacant != NULL);
  
  /* Create mutex semaphore */
  b->mutex = xSemaphoreCreateMutex();
  configASSERT(b->mutex != NULL);
}

/*
 * Blocking insert of ITEM into B.
 */
void buffer_put(Buffer* b, void* item)
{
  /* Wait until at least one vacant slot (block forever) */
  xSemaphoreTake(b->vacant, portMAX_DELAY);
  
  /* Wait for exclusive access to the shared buffer */
  xSemaphoreTake(b->mutex, portMAX_DELAY);

  b->buffer[b->next_in] = item;
  b->next_in = (b->next_in + 1) % b->capacity;

  /* Release mutex */
  xSemaphoreGive(b->mutex);
  
  /* Signal one occupied slot */
  xSemaphoreGive(b->occupied);
}

/*
 * Blocking retrieval of next B entry
 */
void* buffer_get(Buffer* b)
{
  /* Wait until at least one slot has a valid entry */
  xSemaphoreTake(b->occupied, portMAX_DELAY);
  
  /* Wait for exclusive access to the shared buffer */
  xSemaphoreTake(b->mutex, portMAX_DELAY);

  void* item = b->buffer[b->next_out];
  b->next_out = (b->next_out + 1) % b->capacity;

  /* Release mutex */
  xSemaphoreGive(b->mutex);
  
  /* Signal one vacant slot */
  xSemaphoreGive(b->vacant);

  return item;
}

/*
 * Cleanup buffer resources (call when done)
 */
void buffer_cleanup(Buffer* b)
{
  if (b->buffer) {
    vPortFree(b->buffer);
    b->buffer = NULL;
  }
  
  if (b->occupied) {
    vSemaphoreDelete(b->occupied);
    b->occupied = NULL;
  }
  
  if (b->vacant) {
    vSemaphoreDelete(b->vacant);
    b->vacant = NULL;
  }
  
  if (b->mutex) {
    vSemaphoreDelete(b->mutex);
    b->mutex = NULL;
  }
}
