SRCS = hello_c.c hello_c_hostname.c hello_c_hostname_pid.c \
       hello_c_procname.c hello_c_procname_pid.c ring_c.c
OBJS = $(SRCS:.c=.o)
TGTS = $(SRCS:.c=) 

CC	= mpicc
DBG	= -g
OPT	= 
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
