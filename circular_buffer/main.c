/*
 * main.c
 *
 * Copyright (C) 1993 David C. Harrison. All rights reserved.
 *
 * You may not use, distribute, publish, or modify this code without the
 * express written permission of the copyright holder.
 * 
 * ------ 8< ------------------------------------------------------------
 *  
 * Random length random strings are produced by multiple threads to be 
 * consumed by multiple others via a circular bounded buffer. 
 * 
 * Producers are fast, consumers are slow. More consumers than producers
 * requires producers to wait (block) for buffer slots to become available.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>

#include "buffer.h"

#define PRODUCERS    2
#define CONSUMERS    5

#define MAX_SLEEP_MILLISECONDS 600
#define MAX_PUTS_PER_PRODUCER  24

#define BUFFER_SIZE 10

const char charset[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

Buffer buffer;

static void sleep_ms(int msec)
{
  struct timespec ts;
  ts.tv_sec = msec / 1000;
  ts.tv_nsec = (msec % 1000) * 1000000;
  nanosleep(&ts, NULL);
}

char* random_string() 
{
  int length = (rand() % 11) + 5; 
  char *str = malloc(length + 1);
  for (int i = 0; i < length; i++) {
    str[i] = charset[rand() % (sizeof(charset) - 1)];
  }
  str[length] = '\0';
  return str;
}

static void *produce(void *arg)
{
  long tid = (long) arg;
  char* str;
  printf("P %ld running\n", tid);

  for (int i = 0; i < MAX_PUTS_PER_PRODUCER; i++) {
    str = random_string();
    buffer_put(&buffer, str);
    printf("=> P %ld produced %s\n", tid, str);
  }

  printf("P %ld ### finished ###\n", tid);
  pthread_exit(NULL);
  return NULL;
}

static void *consume(void *arg)
{
  long tid = (long) arg;
  int millis;
  char* str;
  printf("C %ld running\n", tid);

  for (;;) {
    millis = rand() % MAX_SLEEP_MILLISECONDS;
    str = buffer_get(&buffer);
    printf("<= C %ld consumed %s - sleeping for %dms\n", tid, str, millis);
    free(str);
    sleep_ms(millis);
  }

  // Will never return, consumers are in an infinite loop
  return NULL;
}

int main(int argc, char* argv[])
{
  buffer_init(&buffer, BUFFER_SIZE);
  pthread_t junk;

  srand(getpid());

  for(long tid = 0; tid < CONSUMERS; tid++) {
    pthread_create(&junk, NULL, consume, (void*)tid);
  }

  for (long tid = 0; tid < PRODUCERS; tid++) {
    pthread_create(&junk, NULL, produce, (void*)tid);
  }

  pthread_exit(NULL);

  // Will never reach here as consumers run forever
  return (EXIT_SUCCESS);
}