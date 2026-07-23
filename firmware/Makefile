# SDF-Edge Firmware Makefile for STM32H743 (Renode simulation)

TOOLCHAIN = C:/Users/maste/Tools/arm-gcc/arm-gnu-toolchain-13.2.Rel1-mingw-w64-i686-arm-none-eabi/bin
CC = $(TOOLCHAIN)/arm-none-eabi-gcc
OBJCOPY = $(TOOLCHAIN)/arm-none-eabi-objcopy
OBJDUMP = $(TOOLCHAIN)/arm-none-eabi-objdump
SIZE = $(TOOLCHAIN)/arm-none-eabi-size

TARGET = sdf_edge_firmware

CFLAGS = -mcpu=cortex-m7 -mthumb -mfloat-abi=hard -mfpu=fpv5-d16
CFLAGS += -O2 -g -Wall -ffunction-sections -fdata-sections
CFLAGS += -DSTM32H743xx -nostdlib

LDFLAGS = -mcpu=cortex-m7 -mthumb -mfloat-abi=hard -mfpu=fpv5-d16
LDFLAGS += -T stm32h743.ld -nostdlib -Wl,--gc-sections -Wl,-Map=$(TARGET).map

SRCS = sdf_edge_firmware.c
OBJS = $(SRCS:.c=.o)

all: $(TARGET).elf $(TARGET).bin
	$(SIZE) $(TARGET).elf

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

$(TARGET).elf: $(OBJS)
	$(CC) $(LDFLAGS) $(OBJS) -o $@ -lgcc

$(TARGET).bin: $(TARGET).elf
	$(OBJCOPY) -O binary $< $@

disasm: $(TARGET).elf
	$(OBJDUMP) -d $< > $(TARGET).dis

clean:
	rm -f $(OBJS) $(TARGET).elf $(TARGET).bin $(TARGET).map $(TARGET).dis

.PHONY: all clean disasm
