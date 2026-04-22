/*
 * buffer.h
 *
 * Circular Bounded Buffer protected by unnamed POSIX semaphores.
 *
 * Copyright (C) 1993-2026 David C. Harrison. All rights reserved.
 *
 * You may not use, distribute, publish, or modify this code without the
 * express written permission of the copyright holder.
 * 
 * Only works on UNIX and UNIX-like systems, so Linux and macOS are fine
 * though you get warnings on macOS. I recommend Linux.
 *
 * If you're on Windows, you have my condolences. ¯\_(ツ)_/¯
 */

#ifndef BUFFER_H
#define BUFFER_H

#include <semaphore.h>

typedef struct buffer_t
{
  void **buffer;   /* shared buffer */
  size_t capacity; /* maximum size of the buffer */
  int next_in;     /* next slot to add an element to (may not currently be vacant) */
  int next_out;    /* next slot to get an element from (may not currently be occupied) */
  sem_t occupied;  /* counting semaphore for occupied slots */
  sem_t vacant;    /* counting semaphore for vacant slots */
  sem_t mutex;     /* binary semaphore to facilitate mutually exclusive access to the shared buffer */
}
Buffer;

void buffer_init(Buffer* b, size_t capacity);
void buffer_put(Buffer* b, void* item);
void* buffer_get(Buffer* b);

#endif // BUFFER_H