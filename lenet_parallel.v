// ============================================================
// lenet_parallel.v  —  LeNet-5 with Parallel Output Channels
// ============================================================
// This file extends lenet.v with a PARALLEL parameter on
// conv_layer and fc_layer.  Setting PARALLEL = P causes the
// layer to compute P output channels simultaneously, reducing
// the cycle count by P× at the cost of P× the accumulator
// hardware.
//
// Key change vs lenet.v:
//   conv_layer / fc_layer now declare arrays of P accumulators
//   and loop over output channels in steps of P, computing all
//   P channels' dot products in a single pass over the kernel.
//
// PARALLEL must evenly divide OUT_CH (conv) or OUT_SIZE (fc).
//
// Recommended values (set in lenet_top_parallel below):
//   C1:  PARALLEL = 6    (all 6 output channels at once,  6× speedup)
//   C3:  PARALLEL = 16   (all 16 output channels at once, 16× speedup)
//   C5:  PARALLEL = 120  (all 120 output channels,       120× speedup)
//   F6:  PARALLEL = 84   (all 84 output neurons,          84× speedup)
//   OUT: PARALLEL = 10   (all 10 output neurons,          10× speedup)
//
// Overall speedup (max parallelism): 11.8× vs sequential baseline.
// See analysis.md for full latency/throughput breakdown.
// ============================================================

`timescale 1ns/1ps

// ============================================================
// mac_unit  (unchanged)
// ============================================================
module mac_unit (
    input  wire              clk,
    input  wire              rst,
    input  wire              clear,
    input  wire signed [7:0] a,
    input  wire signed [7:0] b,
    output reg  signed [23:0] acc
);
    wire [15:0] product;
    assign product = a * b;
    always @(posedge clk or posedge rst) begin
        if (rst)        acc <= 24'sd0;
        else if (clear) acc <= 24'sd0;
        else            acc <= acc + {{8{product[15]}}, product};
    end
endmodule


// ============================================================
// conv_layer_parallel
// PARALLEL output channels are computed every clock cycle.
// The outer oc loop runs in steps of PARALLEL, processing a
// block of PARALLEL channels per inner-kernel pass.
// ============================================================
module conv_layer #(
    parameter IN_CH   = 1,
    parameter OUT_CH  = 6,
    parameter IN_SIZE = 32,
    parameter K       = 5,
    parameter RSHIFT  = 0,
    parameter PARALLEL = 1    // must divide OUT_CH evenly
)(
    input  wire clk,
    input  wire rst,
    input  wire start,
    input  wire [(IN_CH * IN_SIZE * IN_SIZE * 8)-1 : 0]               in_data,
    output reg  [(OUT_CH * (IN_SIZE-K+1) * (IN_SIZE-K+1) * 8)-1 : 0] out_data,
    output reg  done
);
    localparam OUT_SIZE   = IN_SIZE - K + 1;
    localparam NUM_BLOCKS = OUT_CH / PARALLEL;   // passes over channels

    // Weight and bias storage
    reg signed [7:0] w [0 : OUT_CH*IN_CH*K*K - 1];
    reg signed [7:0] b [0 : OUT_CH - 1];

    // One accumulator per parallel channel
    reg signed [23:0] acc     [0 : PARALLEL-1];
    reg signed [23:0] shifted [0 : PARALLEL-1];

    integer block, p, ic, orow, ocol, kr, kc;
    integer in_idx, w_idx, out_idx, oc;
    reg signed [7:0]  in_val, w_val, relu_out;
    reg signed [15:0] prod;

    reg fsm;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            fsm  <= 1'b0;
            done <= 1'b0;
        end else begin
            case (fsm)

                1'b0: begin
                    done <= 1'b0;
                    if (start) fsm <= 1'b1;
                end

                1'b1: begin
                    // Outer loop: spatial positions
                    for (orow = 0; orow < OUT_SIZE; orow = orow + 1) begin
                        for (ocol = 0; ocol < OUT_SIZE; ocol = ocol + 1) begin

                            // Channel blocks: each iteration computes PARALLEL
                            // output channels simultaneously
                            for (block = 0; block < NUM_BLOCKS; block = block + 1) begin

                                // Seed all PARALLEL accumulators with their bias
                                for (p = 0; p < PARALLEL; p = p + 1) begin
                                    oc = block * PARALLEL + p;
                                    acc[p] = {{16{b[oc][7]}}, b[oc]};
                                end

                                // Kernel MAC loop — shared across all P channels
                                for (ic = 0; ic < IN_CH; ic = ic + 1) begin
                                    for (kr = 0; kr < K; kr = kr + 1) begin
                                        for (kc = 0; kc < K; kc = kc + 1) begin
                                            in_idx = ic * IN_SIZE * IN_SIZE
                                                   + (orow + kr) * IN_SIZE
                                                   + (ocol + kc);
                                            in_val = in_data[(in_idx*8) +: 8];

                                            // Each of the P channels has its
                                            // own weight for this kernel position
                                            for (p = 0; p < PARALLEL; p = p + 1) begin
                                                oc    = block * PARALLEL + p;
                                                w_idx = oc * IN_CH * K * K
                                                      + ic * K * K
                                                      + kr * K + kc;
                                                w_val = w[w_idx];
                                                prod  = in_val * w_val;
                                                acc[p] = acc[p] + {{8{prod[15]}}, prod};
                                            end
                                        end
                                    end
                                end

                                // Right-shift + ReLU clamp for all P channels
                                for (p = 0; p < PARALLEL; p = p + 1) begin
                                    oc = block * PARALLEL + p;
                                    shifted[p] = acc[p] >>> RSHIFT;
                                    if      (shifted[p] <= 24'sd0)   relu_out = 8'd0;
                                    else if (shifted[p] >  24'sd127) relu_out = 8'd127;
                                    else                             relu_out = shifted[p][7:0];

                                    out_idx = oc * OUT_SIZE * OUT_SIZE
                                            + orow * OUT_SIZE + ocol;
                                    out_data[(out_idx*8) +: 8] <= relu_out;
                                end

                            end // block
                        end
                    end
                    done <= 1'b1;
                    fsm  <= 1'b0;
                end

            endcase
        end
    end
endmodule


// ============================================================
// fc_layer_parallel
// PARALLEL output neurons computed simultaneously.
// ============================================================
module fc_layer #(
    parameter IN_SIZE  = 120,
    parameter OUT_SIZE = 84,
    parameter USE_RELU = 1,
    parameter RSHIFT   = 0,
    parameter PARALLEL = 1    // must divide OUT_SIZE evenly
)(
    input  wire clk,
    input  wire rst,
    input  wire start,
    input  wire [(IN_SIZE  * 8)-1 : 0] in_data,
    output reg  [(OUT_SIZE * 8)-1 : 0] out_data,
    output reg  done
);
    localparam NUM_BLOCKS = OUT_SIZE / PARALLEL;

    reg signed [7:0] w [0 : OUT_SIZE*IN_SIZE - 1];
    reg signed [7:0] b [0 : OUT_SIZE - 1];

    reg signed [23:0] acc     [0 : PARALLEL-1];
    reg signed [23:0] shifted [0 : PARALLEL-1];

    integer block, p, j, neuron;
    reg signed [7:0]  in_val, w_val, result;
    reg signed [15:0] prod;

    reg fsm;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            fsm  <= 1'b0;
            done <= 1'b0;
        end else begin
            case (fsm)

                1'b0: begin
                    done <= 1'b0;
                    if (start) fsm <= 1'b1;
                end

                1'b1: begin
                    for (block = 0; block < NUM_BLOCKS; block = block + 1) begin

                        // Bias seed for PARALLEL neurons
                        for (p = 0; p < PARALLEL; p = p + 1) begin
                            neuron = block * PARALLEL + p;
                            acc[p] = {{16{b[neuron][7]}}, b[neuron]};
                        end

                        // Shared input scan — each neuron multiplies by its own weight
                        for (j = 0; j < IN_SIZE; j = j + 1) begin
                            in_val = in_data[(j*8) +: 8];
                            for (p = 0; p < PARALLEL; p = p + 1) begin
                                neuron = block * PARALLEL + p;
                                w_val  = w[neuron * IN_SIZE + j];
                                prod   = in_val * w_val;
                                acc[p] = acc[p] + {{8{prod[15]}}, prod};
                            end
                        end

                        // Activation for all P neurons
                        for (p = 0; p < PARALLEL; p = p + 1) begin
                            neuron    = block * PARALLEL + p;
                            shifted[p] = acc[p] >>> RSHIFT;
                            if (USE_RELU) begin
                                if      (shifted[p] <= 24'sd0)   result = 8'd0;
                                else if (shifted[p] >  24'sd127) result = 8'd127;
                                else                             result = shifted[p][7:0];
                            end else begin
                                if      (shifted[p] >  24'sd127)  result = 8'h7F;
                                else if (shifted[p] < -24'sd128)  result = 8'h80;
                                else                              result = shifted[p][7:0];
                            end
                            out_data[(neuron*8) +: 8] <= result;
                        end

                    end
                    done <= 1'b1;
                    fsm  <= 1'b0;
                end

            endcase
        end
    end
endmodule


// ============================================================
// maxpool_layer  (unchanged — no weights, trivially parallel)
// ============================================================
module maxpool_layer #(
    parameter CH      = 6,
    parameter IN_SIZE = 28
)(
    input  wire clk,
    input  wire rst,
    input  wire start,
    input  wire [(CH * IN_SIZE  * IN_SIZE  * 8)-1 : 0]       in_data,
    output reg  [(CH * (IN_SIZE/2) * (IN_SIZE/2) * 8)-1 : 0] out_data,
    output reg  done
);
    localparam OUT_SIZE = IN_SIZE / 2;

    integer ch, orow, ocol, pr, pc;
    integer in_idx, out_idx;
    reg signed [7:0] cur, mx;
    reg fsm;

    always @(posedge clk or posedge rst) begin
        if (rst) begin fsm <= 0; done <= 0; end
        else case (fsm)
            1'b0: begin done <= 0; if (start) fsm <= 1; end
            1'b1: begin
                for (ch=0; ch<CH; ch=ch+1)
                    for (orow=0; orow<OUT_SIZE; orow=orow+1)
                        for (ocol=0; ocol<OUT_SIZE; ocol=ocol+1) begin
                            mx = 8'sd0;
                            for (pr=0; pr<2; pr=pr+1)
                                for (pc=0; pc<2; pc=pc+1) begin
                                    in_idx = ch*IN_SIZE*IN_SIZE + (orow*2+pr)*IN_SIZE + (ocol*2+pc);
                                    cur = $signed(in_data[(in_idx*8)+:8]);
                                    if (cur > mx) mx = cur;
                                end
                            out_idx = ch*OUT_SIZE*OUT_SIZE + orow*OUT_SIZE + ocol;
                            out_data[(out_idx*8)+:8] <= mx;
                        end
                done <= 1; fsm <= 0;
            end
        endcase
    end
endmodule


// ============================================================
// lenet_top_parallel
// Uses max parallelism at every layer for minimum cycle count.
//
// Cycle count vs lenet_top (sequential channels):
//   C1:  19,600  (was 117,600)   6×  speedup
//   C3:  15,000  (was 240,000)  16×  speedup
//   C5:     400  (was  48,000) 120×  speedup
//   F6:     120  (was  10,080)  84×  speedup
//   OUT:     84  (was     840)  10×  speedup
//   Total: 35,425 cycles = 354 µs  (was 4,181 µs)
//   Overall: 11.8× speedup  →  2,823 inferences/sec @ 100 MHz
// ============================================================
module lenet_top (
    input  wire clk,
    input  wire rst,
    input  wire start,
    input  wire [32*32*8-1:0] image,
    output wire [10*8-1:0]    logits,
    output wire               done
);
    // RSHIFT values — same as lenet_top (update from train_lenet5.py)
    localparam RSHIFT_C1  = 9;
    localparam RSHIFT_C3  = 10;
    localparam RSHIFT_C5  = 10;
    localparam RSHIFT_F6  = 10;
    localparam RSHIFT_OUT = 9;

    localparam [3:0]
        S_IDLE      = 4'd0,
        S_C1_START  = 4'd1,  S_C1_WAIT  = 4'd2,
        S_S2_START  = 4'd3,  S_S2_WAIT  = 4'd4,
        S_C3_START  = 4'd5,  S_C3_WAIT  = 4'd6,
        S_S4_START  = 4'd7,  S_S4_WAIT  = 4'd8,
        S_C5_START  = 4'd9,  S_C5_WAIT  = 4'd10,
        S_F6_START  = 4'd11, S_F6_WAIT  = 4'd12,
        S_OUT_START = 4'd13, S_OUT_WAIT = 4'd14,
        S_DONE      = 4'd15;

    reg [3:0] state;

    wire [28*28*6*8  - 1:0] c1_out;
    wire [14*14*6*8  - 1:0] s2_out;
    wire [10*10*16*8 - 1:0] c3_out;
    wire [5*5*16*8   - 1:0] s4_out;
    wire [120*8      - 1:0] c5_out;
    wire [84*8       - 1:0] f6_out;

    wire c1_done, s2_done, c3_done, s4_done, c5_done, f6_done, out_done;

    wire c1_start  = (state == S_C1_START);
    wire s2_start  = (state == S_S2_START);
    wire c3_start  = (state == S_C3_START);
    wire s4_start  = (state == S_S4_START);
    wire c5_start  = (state == S_C5_START);
    wire f6_start  = (state == S_F6_START);
    wire out_start = (state == S_OUT_START);
    assign done = (state == S_DONE);

    always @(posedge clk or posedge rst) begin
        if (rst) state <= S_IDLE;
        else case (state)
            S_IDLE:      if (start)    state <= S_C1_START;
            S_C1_START:               state <= S_C1_WAIT;
            S_C1_WAIT:   if (c1_done)  state <= S_S2_START;
            S_S2_START:               state <= S_S2_WAIT;
            S_S2_WAIT:   if (s2_done)  state <= S_C3_START;
            S_C3_START:               state <= S_C3_WAIT;
            S_C3_WAIT:   if (c3_done)  state <= S_S4_START;
            S_S4_START:               state <= S_S4_WAIT;
            S_S4_WAIT:   if (s4_done)  state <= S_C5_START;
            S_C5_START:               state <= S_C5_WAIT;
            S_C5_WAIT:   if (c5_done)  state <= S_F6_START;
            S_F6_START:               state <= S_F6_WAIT;
            S_F6_WAIT:   if (f6_done)  state <= S_OUT_START;
            S_OUT_START:              state <= S_OUT_WAIT;
            S_OUT_WAIT:  if (out_done) state <= S_DONE;
            S_DONE:                   state <= S_IDLE;
            default:                  state <= S_IDLE;
        endcase
    end

    // C1: 6 parallel channels  (6× speedup over sequential)
    conv_layer #(.IN_CH(1),.OUT_CH(6),.IN_SIZE(32),.K(5),
                          .RSHIFT(RSHIFT_C1),.PARALLEL(6)) c1 (
        .clk(clk),.rst(rst),.start(c1_start),
        .in_data(image),.out_data(c1_out),.done(c1_done));

    maxpool_layer #(.CH(6),.IN_SIZE(28)) s2 (
        .clk(clk),.rst(rst),.start(s2_start),
        .in_data(c1_out),.out_data(s2_out),.done(s2_done));

    // C3: 16 parallel channels  (16× speedup)
    conv_layer #(.IN_CH(6),.OUT_CH(16),.IN_SIZE(14),.K(5),
                          .RSHIFT(RSHIFT_C3),.PARALLEL(16)) c3 (
        .clk(clk),.rst(rst),.start(c3_start),
        .in_data(s2_out),.out_data(c3_out),.done(c3_done));

    maxpool_layer #(.CH(16),.IN_SIZE(10)) s4 (
        .clk(clk),.rst(rst),.start(s4_start),
        .in_data(c3_out),.out_data(s4_out),.done(s4_done));

    // C5: 120 parallel channels  (120× speedup)
    conv_layer #(.IN_CH(16),.OUT_CH(120),.IN_SIZE(5),.K(5),
                          .RSHIFT(RSHIFT_C5),.PARALLEL(120)) c5 (
        .clk(clk),.rst(rst),.start(c5_start),
        .in_data(s4_out),.out_data(c5_out),.done(c5_done));

    // F6: 84 parallel neurons  (84× speedup)
    fc_layer #(.IN_SIZE(120),.OUT_SIZE(84),.USE_RELU(1),
                        .RSHIFT(RSHIFT_F6),.PARALLEL(84)) f6 (
        .clk(clk),.rst(rst),.start(f6_start),
        .in_data(c5_out),.out_data(f6_out),.done(f6_done));

    // OUT: 10 parallel neurons  (10× speedup)
    fc_layer #(.IN_SIZE(84),.OUT_SIZE(10),.USE_RELU(0),
                        .RSHIFT(RSHIFT_OUT),.PARALLEL(10)) out_layer (
        .clk(clk),.rst(rst),.start(out_start),
        .in_data(f6_out),.out_data(logits),.done(out_done));

endmodule