SRCS = jacobi-mpi-block.c
OBJS = $(SRCS:.c=.o)
TGTS = $(SRCS:.c=) 

CC	= mpicc
DBG	= -g
OPT	= -O3 -mavx2 -mfma -march=broadwell
CFLAGS	= $(DBG) $(OPT) 
LDLIBS	= -lm

define make-target
$1: $!.o
	$(CC) $(CFLAGS) -o $@ $<
endef

all: $(TGTS)
	$(foreach t , $(TGTS), $(eval $(call make--target,$(t))))

clean:
	@/bin/rm -f $(OBJS)

realclean: clean
	@/bin/rm -f $(TGTS)
