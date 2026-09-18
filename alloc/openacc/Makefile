# serial : host-only build, pragmas ignored (correctness oracle check on any machine)
# acc    : NVIDIA HPC SDK, offload to the T4 (sm_75)
CC  ?= cc
NVC ?= nvc
UNAME := $(shell uname)
ifeq ($(UNAME),Darwin)
  EXT := dylib
else
  EXT := so
endif
LIB := libparity_acc.$(EXT)

serial: alloc_acc.c
	$(CC) -O2 -ffp-contract=off -fPIC -shared -o $(LIB) alloc_acc.c

acc: alloc_acc.c
	$(NVC) -O2 -acc -gpu=cc75 -Mnofma -Minfo=accel -fPIC -shared -o $(LIB) alloc_acc.c

clean:
	rm -f libparity_acc.so libparity_acc.dylib

.PHONY: serial acc clean
