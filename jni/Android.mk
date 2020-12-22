LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)

LOCAL_C_INCLUDES 	:= "$(LOCAL_PATH)/include"

ifeq ($(TARGET_ARCH_ABI), armeabi-v7a)
LOCAL_MODULE		:= androidOCL_32
endif

ifeq ($(TARGET_ARCH_ABI), arm64-v8a)
LOCAL_MODULE		:= androidOCL_64
endif

LOCAL_SRC_FILES		:= compute_opencl.cpp \
					   execute.cpp

LOCAL_CPP_FEATURES 	:= rtti exceptions
APP_ALLOW_MISSING_DEPS = true

LOCAL_CFLAGS 		+= -O3 -lm -std=c++11 -fopenmp
ifeq ($(APP_OPTIM), debug)
LOCAL_CFLAGS		+= -DDEBUG_VERBOSE
endif

LOCAL_LDFLAGS 		+= -lc -lm -ldl
LOCAL_ARM_MODE 		:=arm

ifeq ($(TARGET_ARCH_ABI), armeabi-v7a)
LOCAL_ARM_NEON		:= true
ARCH_ARM_HAVE_NEON	:= true
LOCAL_CFLAGS 		+= -mfpu=neon -march=armv7-a
LOCAL_CFLAGS		+= -DMALI_LIB_32
endif

ifeq ($(TARGET_ARCH_ABI), arm64-v8a)
LOCAL_ARM_NEON		:= true
ARCH_ARM_HAVE_NEON	:= true
LOCAL_CFLAGS 		+= -mfpu=neon -march=armv8-a
endif

LOCAL_LDLIBS		:= -llog -lz -fopenmp
include $(BUILD_EXECUTABLE)