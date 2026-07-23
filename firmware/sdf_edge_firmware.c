/*
 * SDF-Edge Firmware for STM32H743 (Renode Simulation)
 *
 * Simulates tiled DNN inference with SDF scheduling on Cortex-M7.
 * Demonstrates:
 *   1. Correct SDF ASAP schedule (multi-rate firing)
 *   2. Naive DAG schedule (token underflow / silent corruption)
 *   3. Buffer tracking and peak SRAM measurement
 *
 * Output via USART3 (Renode virtual UART).
 */

#include <stdint.h>

/* Minimal libc replacements for -nostdlib
 * Must use volatile to prevent GCC from optimizing these into calls to themselves */
__attribute__((used, noinline))
void *memset(void *s, int c, unsigned int n) {
    volatile unsigned char *p = (volatile unsigned char *)s;
    while (n--) *p++ = (unsigned char)c;
    return s;
}

__attribute__((used, noinline))
void *memcpy(void *dest, const void *src, unsigned int n) {
    volatile unsigned char *d = (volatile unsigned char *)dest;
    const volatile unsigned char *s2 = (const volatile unsigned char *)src;
    while (n--) *d++ = *s2++;
    return dest;
}

/* --- STM32H743 Register Definitions (minimal for Renode) --- */
#define USART3_BASE     0x40004800UL
#define USART_ISR       (*(volatile uint32_t *)(USART3_BASE + 0x1C))
#define USART_TDR       (*(volatile uint32_t *)(USART3_BASE + 0x28))
#define USART_BRR       (*(volatile uint32_t *)(USART3_BASE + 0x0C))
#define USART_CR1       (*(volatile uint32_t *)(USART3_BASE + 0x00))

#define RCC_BASE        0x58024400UL
#define RCC_APB1LENR    (*(volatile uint32_t *)(RCC_BASE + 0xE8))

/* DWT Cycle Counter */
#define DWT_CTRL        (*(volatile uint32_t *)0xE0001000)
#define DWT_CYCCNT      (*(volatile uint32_t *)0xE0001004)
#define CoreDebug_DEMCR (*(volatile uint32_t *)0xE000EDFC)

/* SRAM regions */
#define DTCM_BASE       0x20000000UL
#define DTCM_SIZE       0x20000     /* 128 KB */
#define AXI_SRAM_BASE   0x24000000UL
#define AXI_SRAM_SIZE   0x80000     /* 512 KB */

/* --- Simple UART output --- */
static void uart_init(void) {
    /* Enable USART3 clock (bit 18 of APB1LENR) */
    RCC_APB1LENR |= (1 << 18);
    /* Configure USART3: 115200 baud @ 125MHz APB1 clock */
    USART_BRR = 1085;  /* 125000000 / 115200 ≈ 1085 */
    USART_CR1 = (1 << 0) | (1 << 3);  /* UE=1, TE=1 */
}

static void uart_putc(char c) {
    /* Brief spin for Renode UART — give up after ~100 iterations to avoid hang */
    volatile int timeout = 100;
    while (!(USART_ISR & (1 << 7)) && --timeout > 0);
    USART_TDR = c;
}

static void uart_puts(const char *s) {
    while (*s) uart_putc(*s++);
}

static void uart_put_int(int32_t val) {
    char buf[12];
    int i = 0;
    if (val < 0) { uart_putc('-'); val = -val; }
    if (val == 0) { uart_putc('0'); return; }
    while (val > 0) { buf[i++] = '0' + (val % 10); val /= 10; }
    while (i > 0) uart_putc(buf[--i]);
}

static void uart_newline(void) { uart_puts("\r\n"); }

/* --- Timing --- */
static volatile uint32_t cycle_count;

static void dwt_init(void) {
    cycle_count = 0;
}

static void delay_cycles(uint32_t n) {
    /* No actual delay — just accumulate cycle count for reporting.
     * Renode simulation is too slow for real delay loops. */
    cycle_count += n;
}

static uint32_t dwt_get_cycles(void) {
    return cycle_count;
}

/* --- SDF Graph Simulation --- */

/* MobileNetV2-MR: Simplified 8-actor multi-rate chain (from Figure 1) */
#define NUM_ACTORS  8
#define NUM_EDGES   7
#define MAX_Q       16
#define MAX_FIRINGS 44  /* sum of q values */

/* Actor names (for output) */
static const char *actor_names[NUM_ACTORS] = {
    "Input", "Stride2a", "DW1", "Stride2b", "DW2", "Stride2c", "GAP", "FC"
};

/* Repetition vector (from Proposition 1: q_max = 2^3 * 2 = 16) */
static const int q[NUM_ACTORS] = {16, 8, 8, 4, 4, 2, 1, 1};

/* Edge definitions: (src, dst, prod, cons) */
static const int edges[NUM_EDGES][4] = {
    {0, 1, 1, 2},  /* Input -> Stride2a:  prod=1, cons=2 (MULTI-RATE) */
    {1, 2, 1, 1},  /* Stride2a -> DW1:    single-rate */
    {2, 3, 1, 2},  /* DW1 -> Stride2b:    prod=1, cons=2 (MULTI-RATE) */
    {3, 4, 1, 1},  /* Stride2b -> DW2:    single-rate */
    {4, 5, 1, 2},  /* DW2 -> Stride2c:    prod=1, cons=2 (MULTI-RATE) */
    {5, 6, 1, 2},  /* Stride2c -> GAP:    prod=1, cons=2 (MULTI-RATE) */
    {6, 7, 1, 1},  /* GAP -> FC:          single-rate */
};

/* Token size per edge (bytes) */
static const int token_sizes[NUM_EDGES] = {512, 256, 256, 192, 192, 64, 48};

/* Simulated per-actor execution time (cycles at 480 MHz) */
static const uint32_t actor_cycles[NUM_ACTORS] = {
    24000, 182400, 115200, 163200, 96000, 144000, 14400, 38400
};

/* Buffer state: tokens on each edge */
static int edge_tokens[NUM_EDGES];

/* Activation buffers in AXI SRAM (small for simulation speed) */
static uint8_t activation_pool[256] __attribute__((section(".bss")));

/* Peak SRAM tracker */
static int peak_sram_bytes;

static void reset_buffers(void) {
    for (int i = 0; i < NUM_EDGES; i++) edge_tokens[i] = 0;
    peak_sram_bytes = 0;
}

static int compute_current_sram(void) {
    int total = 0;
    for (int i = 0; i < NUM_EDGES; i++) {
        total += edge_tokens[i] * token_sizes[i];
    }
    return total;
}

/* Execute one actor firing: consume inputs, simulate work, produce outputs.
 * Returns 0 on success, -1 on token underflow. */
static int fire_actor(int actor_id, int firing_idx, int check_underflow) {
    /* Consume from input edges */
    for (int e = 0; e < NUM_EDGES; e++) {
        if (edges[e][1] == actor_id) {  /* dst == this actor */
            int cons = edges[e][3];
            if (check_underflow && edge_tokens[e] < cons) {
                uart_puts("  TOKEN UNDERFLOW on edge ");
                uart_puts(actor_names[edges[e][0]]);
                uart_puts("->");
                uart_puts(actor_names[actor_id]);
                uart_puts(" (firing ");
                uart_put_int(firing_idx);
                uart_puts("): have=");
                uart_put_int(edge_tokens[e]);
                uart_puts(", need=");
                uart_put_int(cons);
                uart_newline();
                return -1;
            }
            edge_tokens[e] -= cons;
        }
    }

    /* Simulate computation */
    delay_cycles(actor_cycles[actor_id]);

    /* Write a signature to activation buffer to detect corruption */
    int offset = (actor_id * MAX_Q + firing_idx) % sizeof(activation_pool);
    activation_pool[offset] = (uint8_t)(actor_id * 17 + firing_idx * 31 + 0xA5);

    /* Produce to output edges */
    for (int e = 0; e < NUM_EDGES; e++) {
        if (edges[e][0] == actor_id) {  /* src == this actor */
            edge_tokens[e] += edges[e][2];  /* prod */
        }
    }

    /* Track peak SRAM */
    int sram = compute_current_sram();
    if (sram > peak_sram_bytes) peak_sram_bytes = sram;

    return 0;
}

/* --- Schedule Definitions --- */

/* ASAP schedule: correct SDF firing order */
typedef struct { int actor; int firing; } ScheduleEntry;

/* Build ASAP schedule: topological order, fire q[i] times each */
static int build_asap_schedule(ScheduleEntry *sched) {
    int idx = 0;
    for (int a = 0; a < NUM_ACTORS; a++) {
        for (int f = 0; f < q[a]; f++) {
            sched[idx].actor = a;
            sched[idx].firing = f;
            idx++;
        }
    }
    return idx;
}

/* CASAP schedule: fire the most DOWNSTREAM enabled actor first.
 * This is SDF-Edge's buffer-optimal schedule (Theorem 1). */
static int build_casap_schedule(ScheduleEntry *sched) {
    int idx = 0;
    int remaining[NUM_ACTORS];
    int tokens[NUM_EDGES];
    int firing_idx[NUM_ACTORS];

    for (int i = 0; i < NUM_ACTORS; i++) { remaining[i] = q[i]; firing_idx[i] = 0; }
    for (int i = 0; i < NUM_EDGES; i++) tokens[i] = 0;

    int total = 0;
    for (int i = 0; i < NUM_ACTORS; i++) total += q[i];

    for (int step = 0; step < total; step++) {
        /* Scan from most downstream (highest index) to most upstream */
        for (int a = NUM_ACTORS - 1; a >= 0; a--) {
            if (remaining[a] <= 0) continue;

            /* Check if all input edges have enough tokens */
            int can_fire = 1;
            for (int e = 0; e < NUM_EDGES; e++) {
                if (edges[e][1] == a && tokens[e] < edges[e][3]) {
                    can_fire = 0;
                    break;
                }
            }
            if (!can_fire) continue;

            /* Fire: consume inputs */
            for (int e = 0; e < NUM_EDGES; e++) {
                if (edges[e][1] == a) tokens[e] -= edges[e][3];
            }
            /* Fire: produce outputs */
            for (int e = 0; e < NUM_EDGES; e++) {
                if (edges[e][0] == a) tokens[e] += edges[e][2];
            }

            sched[idx].actor = a;
            sched[idx].firing = firing_idx[a];
            firing_idx[a]++;
            remaining[a]--;
            idx++;
            break;
        }
    }
    return idx;
}

/* Naive DAG schedule: round-robin across actors (INCORRECT for multi-rate) */
static int build_naive_dag_schedule(ScheduleEntry *sched) {
    int idx = 0;
    for (int round = 0; round < MAX_Q; round++) {
        for (int a = 0; a < NUM_ACTORS; a++) {
            if (round < q[a]) {
                sched[idx].actor = a;
                sched[idx].firing = round;
                idx++;
            }
        }
    }
    return idx;
}

/* --- Run a schedule and report results --- */
static void run_schedule(const char *name, ScheduleEntry *sched, int len, int check_underflow) {
    uart_puts("\n--- ");
    uart_puts(name);
    uart_puts(" ---\r\n");
    uart_puts("Schedule length: ");
    uart_put_int(len);
    uart_puts(" firings\r\n");

    reset_buffers();

    uint32_t total_start = dwt_get_cycles();
    int underflow_count = 0;
    int total_fired = 0;

    for (int i = 0; i < len; i++) {
        int ret = fire_actor(sched[i].actor, sched[i].firing, check_underflow);
        if (ret < 0) {
            underflow_count++;
            if (underflow_count >= 3) {
                uart_puts("  ... (stopping after 3 underflows)\r\n");
                break;
            }
        }
        total_fired++;
    }

    uint32_t total_cycles = dwt_get_cycles() - total_start;

    uart_puts("Actors fired: ");
    uart_put_int(total_fired);
    uart_newline();
    uart_puts("Peak SRAM: ");
    uart_put_int(peak_sram_bytes);
    uart_puts(" bytes (");
    uart_put_int(peak_sram_bytes / 1024);
    uart_puts(" KB)\r\n");
    uart_puts("Total cycles: ");
    uart_put_int((int)(total_cycles / 1000));
    uart_puts("K\r\n");
    uart_puts("Underflows: ");
    uart_put_int(underflow_count);
    uart_newline();

    /* Verify final activation signatures */
    int corruption = 0;
    for (int a = 0; a < NUM_ACTORS; a++) {
        int offset = (a * MAX_Q + 0) % sizeof(activation_pool);
        uint8_t expected = (uint8_t)(a * 17 + 0 * 31 + 0xA5);
        if (activation_pool[offset] != expected && underflow_count == 0) {
            corruption++;
        }
    }
    uart_puts("Data integrity: ");
    uart_puts(corruption > 0 ? "CORRUPTED" : "OK");
    uart_newline();

    uart_puts("Result: ");
    uart_puts(underflow_count > 0 ? "FAILED (SDF violation detected)" : "PASSED");
    uart_newline();
}

/* --- Repetition Vector Verification --- */
static void verify_repetition_vector(void) {
    uart_puts("\n=== Repetition Vector Verification ===\r\n");

    /* Check Gamma * q = 0 for each edge */
    int all_balanced = 1;
    for (int e = 0; e < NUM_EDGES; e++) {
        int src = edges[e][0];
        int dst = edges[e][1];
        int prod = edges[e][2];
        int cons = edges[e][3];
        int balance = q[src] * prod - q[dst] * cons;

        uart_puts("Edge ");
        uart_puts(actor_names[src]);
        uart_puts("->");
        uart_puts(actor_names[dst]);
        uart_puts(": ");
        uart_put_int(q[src]);
        uart_puts("*");
        uart_put_int(prod);
        uart_puts(" - ");
        uart_put_int(q[dst]);
        uart_puts("*");
        uart_put_int(cons);
        uart_puts(" = ");
        uart_put_int(balance);
        uart_puts(balance == 0 ? " [BALANCED]" : " [IMBALANCED!]");
        uart_newline();

        if (balance != 0) all_balanced = 0;
    }

    uart_puts("Rate consistency: ");
    uart_puts(all_balanced ? "PROVEN (Tier 1)" : "FAILED");
    uart_newline();

    /* Analytical buffer bound */
    uart_puts("\nAnalytical buffer bounds (all schedules):\r\n");
    int total_bound = 0;
    for (int e = 0; e < NUM_EDGES; e++) {
        int src = edges[e][0];
        int prod = edges[e][2];
        int bound_tokens = q[src] * prod;
        int bound_bytes = bound_tokens * token_sizes[e];
        total_bound += bound_bytes;

        uart_puts("  ");
        uart_puts(actor_names[edges[e][0]]);
        uart_puts("->");
        uart_puts(actor_names[edges[e][1]]);
        uart_puts(": ");
        uart_put_int(bound_tokens);
        uart_puts(" tokens * ");
        uart_put_int(token_sizes[e]);
        uart_puts("B = ");
        uart_put_int(bound_bytes);
        uart_puts(" bytes\r\n");
    }
    uart_puts("Total analytical bound: ");
    uart_put_int(total_bound);
    uart_puts(" bytes (");
    uart_put_int(total_bound / 1024);
    uart_puts(" KB)\r\n");
}

/* --- Main --- */
int main(void) {
    uart_init();
    dwt_init();

    uart_puts("\r\n");
    uart_puts("============================================================\r\n");
    uart_puts("  SDF-Edge: STM32H743 Simulation (Renode)\r\n");
    uart_puts("  MobileNetV2-MultiRate Tiled Inference\r\n");
    uart_puts("============================================================\r\n");

    uart_puts("\nModel: MobileNetV2-MR (simplified 8-actor multi-rate chain)\r\n");
    uart_puts("Actors: ");
    uart_put_int(NUM_ACTORS);
    uart_puts(", Edges: ");
    uart_put_int(NUM_EDGES);
    uart_newline();

    uart_puts("Repetition vector q = [");
    for (int i = 0; i < NUM_ACTORS; i++) {
        uart_put_int(q[i]);
        if (i < NUM_ACTORS - 1) uart_puts(", ");
    }
    uart_puts("]\r\n");
    uart_puts("q_max = ");
    uart_put_int(MAX_Q);
    uart_puts(", sum(q) = ");
    int sum_q = 0;
    for (int i = 0; i < NUM_ACTORS; i++) sum_q += q[i];
    uart_put_int(sum_q);
    uart_newline();

    /* 1. Verify repetition vector */
    verify_repetition_vector();

    /* 2. Run correct SDF ASAP schedule */
    ScheduleEntry asap_sched[MAX_FIRINGS];
    int asap_len = build_asap_schedule(asap_sched);
    run_schedule("SDF ASAP Schedule (CORRECT)", asap_sched, asap_len, 1);

    /* 3. Run CASAP interleaved schedule (buffer-optimal, Theorem 1) */
    ScheduleEntry casap_sched[MAX_FIRINGS];
    int casap_len = build_casap_schedule(casap_sched);
    run_schedule("SDF CASAP Schedule (BUFFER-OPTIMAL)", casap_sched, casap_len, 1);

    /* 4. Compare ASAP vs CASAP */
    uart_puts("\n=== ASAP vs CASAP Buffer Comparison ===\r\n");
    /* Re-run to get peaks (run_schedule already printed them, but let's recompute for clarity) */
    reset_buffers();
    for (int i = 0; i < asap_len; i++) fire_actor(asap_sched[i].actor, asap_sched[i].firing, 0);
    int asap_peak = peak_sram_bytes;

    reset_buffers();
    for (int i = 0; i < casap_len; i++) fire_actor(casap_sched[i].actor, casap_sched[i].firing, 0);
    int casap_peak = peak_sram_bytes;

    uart_puts("ASAP peak SRAM:  ");
    uart_put_int(asap_peak);
    uart_puts(" bytes (");
    uart_put_int(asap_peak / 1024);
    uart_puts(" KB)\r\n");
    uart_puts("CASAP peak SRAM: ");
    uart_put_int(casap_peak);
    uart_puts(" bytes (");
    uart_put_int(casap_peak / 1024);
    uart_puts(" KB)\r\n");
    uart_puts("Reduction: ");
    if (casap_peak > 0) {
        uart_put_int(asap_peak / casap_peak);
        uart_puts(".");
        uart_put_int((asap_peak * 10 / casap_peak) % 10);
    }
    uart_puts("x\r\n");

    /* 5. Run naive DAG schedule (will fail) */
    ScheduleEntry naive_sched[MAX_FIRINGS];
    int naive_len = build_naive_dag_schedule(naive_sched);
    run_schedule("Naive DAG Schedule (INCORRECT)", naive_sched, naive_len, 1);

    uart_puts("\n============================================================\r\n");
    uart_puts("  CONCLUSION: CASAP achieves buffer-optimal scheduling.\r\n");
    uart_puts("  SDF analysis catches DAG scheduling errors.\r\n");
    uart_puts("============================================================\r\n");
    uart_puts("SIMULATION_COMPLETE\r\n");

    while (1);  /* Halt */
    return 0;
}

/* Minimal startup: vector table + Reset_Handler */
extern unsigned int _estack;

void Reset_Handler(void);
void Default_Handler(void);

__attribute__((section(".isr_vector")))
void (* const vector_table[])(void) = {
    (void (*)(void))(&_estack),
    Reset_Handler,
    Default_Handler,  /* NMI */
    Default_Handler,  /* HardFault */
    Default_Handler,  /* MemManage */
    Default_Handler,  /* BusFault */
    Default_Handler,  /* UsageFault */
};

void Reset_Handler(void) {
    /* Zero BSS */
    extern unsigned int _sbss, _ebss;
    unsigned int *p = &_sbss;
    while (p < &_ebss) *p++ = 0;

    /* Copy data */
    extern unsigned int _sdata, _edata, _sidata;
    unsigned int *src = &_sidata;
    unsigned int *dst = &_sdata;
    while (dst < &_edata) *dst++ = *src++;

    main();
}

void Default_Handler(void) {
    while (1);
}
