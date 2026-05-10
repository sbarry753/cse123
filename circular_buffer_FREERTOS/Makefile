#
# Makefile
#
# Circular Bounded Buffer protected by unnamed POSIX semaphores.
#
# Copyright (C) 1993 David C. Harrison. All rights reserved.
#
# You may not use, distribute, publish, or modify this code without the
# express written permission of the copyright holder.
#
# Only works on UNIX and UNIX-like systems, so Linux and macOS are fine
# though you get warnings on macOS. I recommend Linux.
#
# If you're on Windows, you have my condolences. ¯\_(ツ)_/¯
#
# Only modified for FreeRTOS

CC = gcc
CFLAGS = -Wall

all: main

main: main.o buffer.o
	$(CC) $(CFLAGS) -o $@ $^

main.o: main.c buffer.h
	$(CC) $(CFLAGS) -c $<

buffer.o: buffer.c buffer.h
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f main main.o buffer.o
