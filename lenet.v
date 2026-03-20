// ============================================================
// lenet.v  —  LeNet-5 Neural Network Hardware Accelerator
// ============================================================
// Architecture (8-bit signed integer datapath):
//
//   image (32×32×1, int8)
//     → C1  conv_layer  (IN_CH=1,  OUT_CH=6,   IN_SIZE=32, K=5) → 28×28×6
//     → S2  maxpool_layer (CH=6,   IN_SIZE=28, pool=2×2)        → 14×14×6
//     → C3  conv_layer  (IN_CH=6,  OUT_CH=16,  IN_SIZE=14, K=5) → 10×10×16
//     → S4  maxpool_layer (CH=16,  IN_SIZE=10, pool=2×2)        → 5×5×16
//     → C5  conv_layer  (IN_CH=16, OUT_CH=120, IN_SIZE=5,  K=5) → 1×1×120
//     → F6  fc_layer    (IN=120, OUT=84,  USE_RELU=1)
//     → OUT fc_layer    (IN=84,  OUT=10,  USE_RELU=0)
//     → logits [10]  (raw class scores, no softmax)
//
// RSHIFT parameter (added to conv_layer and fc_layer):
//   After all MACs are summed, the 24-bit accumulator is
//   arithmetically right-shifted by RSHIFT bits before the
//   ReLU / saturation clamp.  This is necessary when using
//   quantized int8 weights whose scales are large (~200–400×),
//   otherwise the accumulated sums always saturate to ±127.
//
//   RSHIFT values are computed by train_lenet5.py via an integer
//   forward-pass simulation and printed at the end of training.
//   The values in lenet_top below match a 5-epoch MNIST run with
//   seed 42; re-run the Python script and update if you retrain.
//
// Feature-map flat-index convention:
//   element(ch, row, col) at bit offset
//     (ch * SIZE * SIZE  +  row * SIZE  +  col) * 8
//
// Weight flat-index convention for conv_layer:
//   element(oc, ic, kr, kc)  →  oc*IN_CH*K*K + ic*K*K + kr*K + kc
//
// Weight flat-index convention for fc_layer:
//   element(out_neuron, in_neuron)  →  out_neuron * IN_SIZE + in_neuron
// ============================================================

`timescale 1ns/1ps

// ============================================================
// mac_unit
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
// relu8
// ============================================================
module relu8 (
    input  wire signed [23:0] in_val,
    output wire        [7:0]  out_val
);
    assign out_val = (in_val <= 24'sd0)   ? 8'd0  :
                     (in_val >  24'sd127) ? 8'd127 :
                     in_val[7:0];
endmodule


// ============================================================
// conv_layer
// Parameters:
//   IN_CH    input channels
//   OUT_CH   output channels (filters)
//   IN_SIZE  spatial input dimension (square)
//   K        kernel size (square)
//   RSHIFT   arithmetic right-shift applied to the 24-bit
//            accumulator before the saturating ReLU clamp.
//            train_lenet5.py prints the correct value for your
//            trained model.
// ============================================================
module conv_layer #(
    parameter IN_CH   = 1,
    parameter OUT_CH  = 6,
    parameter IN_SIZE = 32,
    parameter K       = 5,
    parameter RSHIFT  = 0      // ← set by lenet_top from Python output
)(
    input  wire clk,
    input  wire rst,
    input  wire start,
    input  wire [(IN_CH * IN_SIZE * IN_SIZE * 8)-1 : 0]               in_data,
    output reg  [(OUT_CH * (IN_SIZE-K+1) * (IN_SIZE-K+1) * 8)-1 : 0] out_data,
    output reg  done
);
    localparam OUT_SIZE = IN_SIZE - K + 1;

    reg signed [7:0] w [0 : OUT_CH*IN_CH*K*K - 1];
    reg signed [7:0] b [0 : OUT_CH - 1];

    integer oc, ic, orow, ocol, kr, kc;
    integer in_idx, w_idx, out_idx;

    reg signed [7:0]  in_val, w_val, relu_out;
    reg signed [15:0] prod;
    reg signed [23:0] acc, shifted;

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
                    for (oc = 0; oc < OUT_CH; oc = oc + 1) begin
                        for (orow = 0; orow < OUT_SIZE; orow = orow + 1) begin
                            for (ocol = 0; ocol < OUT_SIZE; ocol = ocol + 1) begin

                                acc = {{16{b[oc][7]}}, b[oc]};

                                for (ic = 0; ic < IN_CH; ic = ic + 1) begin
                                    for (kr = 0; kr < K; kr = kr + 1) begin
                                        for (kc = 0; kc < K; kc = kc + 1) begin
                                            in_idx = ic * IN_SIZE * IN_SIZE
                                                   + (orow + kr) * IN_SIZE
                                                   + (ocol + kc);
                                            w_idx  = oc * IN_CH * K * K
                                                   + ic * K * K
                                                   + kr * K + kc;
                                            in_val = in_data[(in_idx*8) +: 8];
                                            w_val  = w[w_idx];
                                            prod   = in_val * w_val;
                                            acc    = acc + {{8{prod[15]}}, prod};
                                        end
                                    end
                                end

                                // ── Right-shift then saturating ReLU ──
                                shifted = acc >>> RSHIFT;
                                if      (shifted <= 24'sd0)   relu_out = 8'd0;
                                else if (shifted >  24'sd127) relu_out = 8'd127;
                                else                          relu_out = shifted[7:0];

                                out_idx = oc * OUT_SIZE * OUT_SIZE
                                        + orow * OUT_SIZE + ocol;
                                out_data[(out_idx*8) +: 8] <= relu_out;
                            end
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
// maxpool_layer  (unchanged — no weights, no scaling needed)
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
                    for (ch = 0; ch < CH; ch = ch + 1) begin
                        for (orow = 0; orow < OUT_SIZE; orow = orow + 1) begin
                            for (ocol = 0; ocol < OUT_SIZE; ocol = ocol + 1) begin
                                mx = 8'sd0;
                                for (pr = 0; pr < 2; pr = pr + 1) begin
                                    for (pc = 0; pc < 2; pc = pc + 1) begin
                                        in_idx = ch * IN_SIZE * IN_SIZE
                                               + (orow*2 + pr) * IN_SIZE
                                               + (ocol*2 + pc);
                                        cur = $signed(in_data[(in_idx*8) +: 8]);
                                        if (cur > mx) mx = cur;
                                    end
                                end
                                out_idx = ch * OUT_SIZE * OUT_SIZE
                                        + orow * OUT_SIZE + ocol;
                                out_data[(out_idx*8) +: 8] <= mx;
                            end
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
// fc_layer
// Parameters:
//   IN_SIZE   input neurons
//   OUT_SIZE  output neurons
//   USE_RELU  1 → ReLU clamp [0,127]   0 → signed sat [-128,127]
//   RSHIFT    arithmetic right-shift before activation clamp
// ============================================================
module fc_layer #(
    parameter IN_SIZE  = 120,
    parameter OUT_SIZE = 84,
    parameter USE_RELU = 1,
    parameter RSHIFT   = 0     // ← set by lenet_top from Python output
)(
    input  wire clk,
    input  wire rst,
    input  wire start,
    input  wire [(IN_SIZE  * 8)-1 : 0] in_data,
    output reg  [(OUT_SIZE * 8)-1 : 0] out_data,
    output reg  done
);
    reg signed [7:0] w [0 : OUT_SIZE*IN_SIZE - 1];
    reg signed [7:0] b [0 : OUT_SIZE - 1];

    integer i, j;
    reg signed [7:0]  in_val, w_val, result;
    reg signed [15:0] prod;
    reg signed [23:0] acc, shifted;

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
                    for (i = 0; i < OUT_SIZE; i = i + 1) begin
                        acc = {{16{b[i][7]}}, b[i]};
                        for (j = 0; j < IN_SIZE; j = j + 1) begin
                            in_val = in_data[(j*8) +: 8];
                            w_val  = w[i*IN_SIZE + j];
                            prod   = in_val * w_val;
                            acc    = acc + {{8{prod[15]}}, prod};
                        end

                        // ── Right-shift then activation ──
                        shifted = acc >>> RSHIFT;
                        if (USE_RELU) begin
                            if      (shifted <= 24'sd0)   result = 8'd0;
                            else if (shifted >  24'sd127) result = 8'd127;
                            else                          result = shifted[7:0];
                        end else begin
                            if      (shifted >  24'sd127)  result = 8'h7F;
                            else if (shifted < -24'sd128)  result = 8'h80;
                            else                           result = shifted[7:0];
                        end
                        out_data[(i*8) +: 8] <= result;
                    end
                    done <= 1'b1;
                    fsm  <= 1'b0;
                end

            endcase
        end
    end
endmodule


// ============================================================
// lenet_top
//
// RSHIFT values below are computed by train_lenet5.py.
// After running the Python script, look for the block:
//   "Paste these RSHIFT values into lenet_top:"
// and replace the five RSHIFT lines with the printed values.
// ============================================================
module lenet_top (
    input  wire clk,
    input  wire rst,
    input  wire start,
    input  wire [32*32*8-1:0] image,
    output wire [10*8-1:0]    logits,
    output wire               done
);

    // ── RSHIFT values (update from train_lenet5.py output) ──────
    localparam RSHIFT_C1  = 9;
    localparam RSHIFT_C3  = 10;
    localparam RSHIFT_C5  = 10;
    localparam RSHIFT_F6  = 10;
    localparam RSHIFT_OUT = 9;

    // ── FSM ──────────────────────────────────────────────────────
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

    wire [28*28*6*8  - 1 : 0] c1_out;
    wire [14*14*6*8  - 1 : 0] s2_out;
    wire [10*10*16*8 - 1 : 0] c3_out;
    wire [5*5*16*8   - 1 : 0] s4_out;
    wire [120*8      - 1 : 0] c5_out;
    wire [84*8       - 1 : 0] f6_out;

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
        if (rst)
            state <= S_IDLE;
        else
            case (state)
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

    conv_layer #(.IN_CH(1),  .OUT_CH(6),   .IN_SIZE(32), .K(5), .RSHIFT(RSHIFT_C1))  c1 (
        .clk(clk), .rst(rst), .start(c1_start),
        .in_data(image),   .out_data(c1_out), .done(c1_done));

    maxpool_layer #(.CH(6),  .IN_SIZE(28)) s2 (
        .clk(clk), .rst(rst), .start(s2_start),
        .in_data(c1_out),  .out_data(s2_out), .done(s2_done));

    conv_layer #(.IN_CH(6),  .OUT_CH(16),  .IN_SIZE(14), .K(5), .RSHIFT(RSHIFT_C3))  c3 (
        .clk(clk), .rst(rst), .start(c3_start),
        .in_data(s2_out),  .out_data(c3_out), .done(c3_done));

    maxpool_layer #(.CH(16), .IN_SIZE(10)) s4 (
        .clk(clk), .rst(rst), .start(s4_start),
        .in_data(c3_out),  .out_data(s4_out), .done(s4_done));

    conv_layer #(.IN_CH(16), .OUT_CH(120), .IN_SIZE(5),  .K(5), .RSHIFT(RSHIFT_C5))  c5 (
        .clk(clk), .rst(rst), .start(c5_start),
        .in_data(s4_out),  .out_data(c5_out), .done(c5_done));

    fc_layer #(.IN_SIZE(120), .OUT_SIZE(84), .USE_RELU(1), .RSHIFT(RSHIFT_F6))  f6 (
        .clk(clk), .rst(rst), .start(f6_start),
        .in_data(c5_out),  .out_data(f6_out), .done(f6_done));

    fc_layer #(.IN_SIZE(84), .OUT_SIZE(10), .USE_RELU(0), .RSHIFT(RSHIFT_OUT)) out_layer (
        .clk(clk), .rst(rst), .start(out_start),
        .in_data(f6_out),  .out_data(logits), .done(out_done));

endmodule