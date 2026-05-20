#ifndef BUFFER_H
#define BUFFER_H

#include "FreeRTOS.h"
#include "semphr.h"
#include <stddef.h>

typedef struct buffer_t
{
  void **buffer;              /* shared buffer */
  size_t capacity;            /* maximum size of the buffer */
  int next_in;                /* next slot to add an element to */
  int next_out;               /* next slot to get an element from */
  SemaphoreHandle_t occupied; /* counting semaphore for occupied slots */
  SemaphoreHandle_t vacant;   /* counting semaphore for vacant slots */
  SemaphoreHandle_t mutex;    /* mutex for exclusive access */
}
Buffer;

void buffer_init(Buffer* b, size_t capacity);
void buffer_put(Buffer* b, void* item);
void* buffer_get(Buffer* b);
void buffer_cleanup(Buffer* b); /* Added cleanup function */

#endif // BUFFER_H
