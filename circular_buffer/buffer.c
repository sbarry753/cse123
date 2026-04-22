/*
 * buffer.c
 *
 * Circular Bounded Buffer protected by unnamed POSIX semaphores.
 *
 * Copyright (C) 1993 David C. Harrison. All rights reserved.
 *
 * You may not use, distribute, publish, or modify this code without the
 * express written permission of the copyright holder.
 *
 * Only works on UNIX and UNIX-like systems, so Linux and macOS are fine
 * though you get warnings on macOS. I recommend Linux.
 *
 * If you're on Windows, you have my condolences. ¯\_(ツ)_/¯
 */

#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>

#include "buffer.h"

/*
 * Initialise the circular bounded buffer B 
 */
void buffer_init(Buffer* b, size_t capacity)
{
  b->capacity = capacity;
  b->buffer = (void**) malloc(b->capacity * sizeof(void*));
  if (!b->buffer) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(EXIT_FAILURE);
  }
  sem_init(&b->occupied, 0, 0);
  sem_init(&b->vacant, 0, b->capacity);
  sem_init(&b->mutex, 0, 1);
}

/*
 * Blocking insert of ITEM into B.
 */
void buffer_put(Buffer* b, void* item)
{
  sem_wait(&b->vacant);   // wait until at least one vacant slot
  sem_wait(&b->mutex);    // wait for exclusive access to the shared buffer

  b->buffer[b->next_in] = item;
  b->next_in = (b->next_in+1) % b->capacity;

  sem_post(&b->mutex);    // release shared buffer so other threads can use it
  sem_post(&b->occupied); // decrement the number of occupied slots
}

/*
 * Blocking retrieval of next B entry
 */
void* buffer_get(Buffer* b)
{
  sem_wait(&b->occupied); // wait until at least one slot has a valid entry
  sem_wait(&b->mutex);    // wait for exclusive access to the shared buffer

  void* item = b->buffer[b->next_out];
  b->next_out = (b->next_out+1) % b->capacity;

  sem_post(&b->mutex);    // release the shared buffer so other threads can use it
  sem_post(&b->vacant);   // decrement the number of vacant slots

  return item;
}