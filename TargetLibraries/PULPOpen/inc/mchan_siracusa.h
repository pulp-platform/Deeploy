// Default mchan base address
#ifndef MCHAN_BASE_ADDR
#define MCHAN_BASE_ADDR (ARCHI_MCHAN_DEMUX_ADDR) // CLUSTER_MCHAN_ADDR
#endif

// Default mchan await mode
#if !defined(MCHAN_EVENT) && !defined(MCHAN_POLLED)
#define MCHAN_EVENT
#endif

#ifdef MCHAN_EVENT
#define MCHAN_EVENT_BIT (ARCHI_CL_EVT_DMA0) // 8
#endif

#include "mchan_v7.h"
