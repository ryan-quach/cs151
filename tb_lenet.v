// ============================================================
// tb_lenet_combined.v  —  LeNet-5 Combined Testbench
// ============================================================
// PHASE 1: 45 unit tests confirming each module's arithmetic
// PHASE 2: Full system test with TRAINED weights from Python
//
// EDA Playground setup:
//   Simulator : Icarus Verilog
//   Design    : lenet.v
//   Testbench : tb_lenet_combined.v
//
// Before running with real weights:
//   1. Run train_lenet5.py on your machine
//   2. Upload all .hex files from weights/ into EDA Playground
//      (left panel -> "+" icon to add files)
//      Files: c1_w c1_b c3_w c3_b c5_w_0 c5_w_1 c5_b f6_w f6_b out_w out_b
//             test_image  (12 files total; c5 is split in two)
//   3. Set TRAINED_WEIGHTS = 1 below
//   4. Set EXPECTED_CLASS  = the digit in test_image_label.txt
//
// If hex files are not available, leave TRAINED_WEIGHTS = 0
// to fall back to pseudo-random weights (system test still runs,
// classification result is meaningless but hardware is exercised).
// ============================================================

`timescale 1ns/1ps

module tb_lenet_combined;

    // ==========================================================
    // ---- CONFIGURATION  (edit these two lines) ---------------
    // ==========================================================
    parameter TRAINED_WEIGHTS = 1;   // set to 1 after uploading hex files
    parameter EXPECTED_CLASS  = 7;   // digit in test_image_label.txt

    // ==========================================================
    // ---- Shared clock / reset --------------------------------
    // ==========================================================
    reg clk, rst;
    initial clk = 1'b0;
    always #5 clk = ~clk;

    // ==========================================================
    // ---- Pass / fail counters --------------------------------
    // ==========================================================
    integer tests_run, tests_passed;
    initial begin tests_run = 0; tests_passed = 0; end

    // ==========================================================
    // ---- check8 : verify one signed 8-bit output ------------
    // ==========================================================
    task check8;
        input [63:0]       test_num;
        input [127:0]      label;
        input signed [7:0] got;
        input signed [7:0] expected;
        begin
            tests_run = tests_run + 1;
            if (got === expected) begin
                $display("  [T%0d] PASS  %-16s  got=%4d  expected=%4d",
                         test_num, label, got, expected);
                tests_passed = tests_passed + 1;
            end else begin
                $display("  [T%0d] FAIL  %-16s  got=%4d  expected=%4d  *** MISMATCH ***",
                         test_num, label, got, expected);
            end
        end
    endtask

    // ==========================================================
    // ---- UNIT TEST FIXTURES ----------------------------------
    // ==========================================================

    // Fixture A: conv_layer  1ch->1ch, 5x5 input, 3x3 kernel
    localparam CA_IN_CH=1, CA_OUT_CH=1, CA_IN_SIZE=5, CA_K=3;
    localparam CA_OUT_SIZE = CA_IN_SIZE - CA_K + 1;

    reg  [(CA_IN_CH*CA_IN_SIZE*CA_IN_SIZE*8)-1:0]           ca_in;
    wire [(CA_OUT_CH*CA_OUT_SIZE*CA_OUT_SIZE*8)-1:0]         ca_out;
    reg  ca_start;
    wire ca_done;

    conv_layer #(.IN_CH(CA_IN_CH),.OUT_CH(CA_OUT_CH),
                 .IN_SIZE(CA_IN_SIZE),.K(CA_K)) ca_dut (
        .clk(clk),.rst(rst),.start(ca_start),
        .in_data(ca_in),.out_data(ca_out),.done(ca_done)
    );

    // Fixture B: maxpool_layer  1ch, 4x4 -> 2x2
    localparam MB_CH=1, MB_IN_SIZE=4;
    localparam MB_OUT_SIZE = MB_IN_SIZE / 2;

    reg  [(MB_CH*MB_IN_SIZE*MB_IN_SIZE*8)-1:0]              mb_in;
    wire [(MB_CH*MB_OUT_SIZE*MB_OUT_SIZE*8)-1:0]             mb_out;
    reg  mb_start;
    wire mb_done;

    maxpool_layer #(.CH(MB_CH),.IN_SIZE(MB_IN_SIZE)) mb_dut (
        .clk(clk),.rst(rst),.start(mb_start),
        .in_data(mb_in),.out_data(mb_out),.done(mb_done)
    );

    // Fixture C: fc_layer  USE_RELU=1, IN=4, OUT=2
    localparam FC_IN=4, FC_OUT=2;

    reg  [(FC_IN*8)-1:0]   fc_in;
    wire [(FC_OUT*8)-1:0]  fc_out_relu;
    reg  fc_relu_start;
    wire fc_relu_done;

    fc_layer #(.IN_SIZE(FC_IN),.OUT_SIZE(FC_OUT),.USE_RELU(1)) fc_relu_dut (
        .clk(clk),.rst(rst),.start(fc_relu_start),
        .in_data(fc_in),.out_data(fc_out_relu),.done(fc_relu_done)
    );

    // Fixture D: fc_layer  USE_RELU=0, IN=4, OUT=2
    wire [(FC_OUT*8)-1:0]  fc_out_noact;
    reg  fc_noact_start;
    wire fc_noact_done;

    fc_layer #(.IN_SIZE(FC_IN),.OUT_SIZE(FC_OUT),.USE_RELU(0)) fc_noact_dut (
        .clk(clk),.rst(rst),.start(fc_noact_start),
        .in_data(fc_in),.out_data(fc_out_noact),.done(fc_noact_done)
    );

    // Fixture E: fc_layer  USE_RELU=0, IN=4, OUT=1  (saturation)
    reg  [(FC_IN*8)-1:0]  fe_in;
    wire [7:0]             fe_out;
    reg  fe_start;
    wire fe_done;

    fc_layer #(.IN_SIZE(FC_IN),.OUT_SIZE(1),.USE_RELU(0)) fe_dut (
        .clk(clk),.rst(rst),.start(fe_start),
        .in_data(fe_in),.out_data(fe_out),.done(fe_done)
    );

    // ==========================================================
    // ---- FULL SYSTEM DUT -------------------------------------
    // ==========================================================
    reg  [32*32*8-1:0] image;
    wire [10*8-1:0]    logits;
    reg  sys_start;
    wire sys_done;

    lenet_top dut (
        .clk(clk), .rst(rst),
        .start(sys_start),
        .image(image),
        .logits(logits),
        .done(sys_done)
    );

    // Temporary flat memory for loading image via $readmemh
    reg [7:0] image_mem [0:1023];

    // ==========================================================
    // ---- HELPERS ---------------------------------------------
    // ==========================================================
    integer _i;

    task fill_flat;
        input [1:0]        target;
        input signed [7:0] val;
        begin
            case (target)
                2'd0: for (_i=0;_i<25;_i=_i+1) ca_in[(_i*8)+:8] = val;
                2'd1: for (_i=0;_i<16;_i=_i+1) mb_in[(_i*8)+:8] = val;
                2'd2: for (_i=0;_i< 4;_i=_i+1) fc_in[(_i*8)+:8] = val;
                2'd3: for (_i=0;_i< 4;_i=_i+1) fe_in[(_i*8)+:8] = val;
            endcase
        end
    endtask

    task pulse_and_wait;
        input [2:0] target;
        begin
            @(negedge clk);
            case (target)
                3'd0: ca_start       = 1;
                3'd1: mb_start       = 1;
                3'd2: fc_relu_start  = 1;
                3'd3: fc_noact_start = 1;
                3'd4: fe_start       = 1;
            endcase
            @(negedge clk);
            case (target)
                3'd0: ca_start       = 0;
                3'd1: mb_start       = 0;
                3'd2: fc_relu_start  = 0;
                3'd3: fc_noact_start = 0;
                3'd4: fe_start       = 0;
            endcase
            case (target)
                3'd0: @(posedge ca_done);
                3'd1: @(posedge mb_done);
                3'd2: @(posedge fc_relu_done);
                3'd3: @(posedge fc_noact_done);
                3'd4: @(posedge fe_done);
            endcase
            @(posedge clk);
        end
    endtask

    // ==========================================================
    // ---- WEIGHT / IMAGE LOADING TASKS ------------------------
    // ==========================================================

    task load_trained_weights;
        begin
            $readmemh("c1_w.hex",  dut.c1.w);
            $readmemh("c1_b.hex",  dut.c1.b);
            $readmemh("c3_w.hex",  dut.c3.w);
            $readmemh("c3_b.hex",  dut.c3.b);
            // c5 weights are split into two files to stay under EDA Playground's
            // 100 000-character per-file limit. Load each half with an address range.
            $readmemh("c5_w_0.hex", dut.c5.w,      0, 23999);
            $readmemh("c5_w_1.hex", dut.c5.w, 24000, 47999);
            $readmemh("c5_b.hex",  dut.c5.b);
            $readmemh("f6_w.hex",  dut.f6.w);
            $readmemh("f6_b.hex",  dut.f6.b);
            $readmemh("out_w.hex", dut.out_layer.w);
            $readmemh("out_b.hex", dut.out_layer.b);
        end
    endtask

    task load_random_weights;
        integer k;
        begin
            for (k=0; k<150;   k=k+1) dut.c1.w[k]        = ($random % 2);
            for (k=0; k<6;     k=k+1) dut.c1.b[k]        = 8'sd0;
            for (k=0; k<2400;  k=k+1) dut.c3.w[k]        = ($random % 2);
            for (k=0; k<16;    k=k+1) dut.c3.b[k]        = 8'sd0;
            for (k=0; k<48000; k=k+1) dut.c5.w[k]        = ($random % 2);
            for (k=0; k<120;   k=k+1) dut.c5.b[k]        = 8'sd0;
            for (k=0; k<10080; k=k+1) dut.f6.w[k]        = ($random % 2);
            for (k=0; k<84;    k=k+1) dut.f6.b[k]        = 8'sd0;
            for (k=0; k<840;   k=k+1) dut.out_layer.w[k] = ($random % 2);
            for (k=0; k<10;    k=k+1) dut.out_layer.b[k] = 8'sd0;
        end
    endtask

    task load_test_image;
        integer px;
        begin
            $readmemh("test_image.hex", image_mem);
            for (px=0; px<1024; px=px+1)
                image[(px*8)+:8] = image_mem[px];
        end
    endtask

    task load_gradient_image;
        integer r, c;
        begin
            for (r=0; r<32; r=r+1)
                for (c=0; c<32; c=c+1)
                    image[((r*32+c)*8)+:8] = (r+c) % 50;
        end
    endtask

    // ==========================================================
    // ---- MAIN SEQUENCE ---------------------------------------
    // ==========================================================
    integer p, i, nonzero, predicted;
    reg signed [7:0] best_logit;
    integer sys_pass;
    initial sys_pass = 0;

    initial begin
        $dumpfile("dump.vcd");
        $dumpvars(0, tb_lenet_combined);

        ca_start = 0; mb_start = 0;
        fc_relu_start = 0; fc_noact_start = 0; fe_start = 0;
        sys_start = 0;

        rst = 1;
        repeat(4) @(posedge clk);
        rst = 0;
        @(posedge clk);

        // ==================================================
        // PHASE 1 - UNIT TESTS
        // ==================================================
        $display("");
        $display("======================================================");
        $display("  PHASE 1: Module Unit Tests");
        $display("======================================================");

        $display("\nT1: conv_layer  basic  (px=2, w=1, b=0 -> 18)");
        fill_flat(0, 8'sd2);
        for (_i=0; _i < CA_OUT_CH*CA_IN_CH*CA_K*CA_K; _i=_i+1)
            ca_dut.w[_i] = 8'sd1;
        for (_i=0; _i < CA_OUT_CH; _i=_i+1)
            ca_dut.b[_i] = 8'sd0;
        pulse_and_wait(0);
        for (p=0; p < CA_OUT_SIZE*CA_OUT_SIZE; p=p+1)
            check8(1, "conv_out_px", $signed(ca_out[(p*8)+:8]), 8'sd18);

        $display("\nT2: conv_layer  ReLU clips  (px=2, w=-1, b=0 -> 0)");
        for (_i=0; _i < CA_OUT_CH*CA_IN_CH*CA_K*CA_K; _i=_i+1)
            ca_dut.w[_i] = -8'sd1;
        pulse_and_wait(0);
        for (p=0; p < CA_OUT_SIZE*CA_OUT_SIZE; p=p+1)
            check8(2, "conv_relu0", $signed(ca_out[(p*8)+:8]), 8'sd0);

        $display("\nT3: conv_layer  bias  (px=1, w=1, b=10 -> 19)");
        fill_flat(0, 8'sd1);
        for (_i=0; _i < CA_OUT_CH*CA_IN_CH*CA_K*CA_K; _i=_i+1)
            ca_dut.w[_i] = 8'sd1;
        ca_dut.b[0] = 8'sd10;
        pulse_and_wait(0);
        for (p=0; p < CA_OUT_SIZE*CA_OUT_SIZE; p=p+1)
            check8(3, "conv_bias", $signed(ca_out[(p*8)+:8]), 8'sd19);

        $display("\nT4: conv_layer  sat  (px=127, w=3, acc=3429 -> 127)");
        fill_flat(0, 8'sd127);
        for (_i=0; _i < CA_OUT_CH*CA_IN_CH*CA_K*CA_K; _i=_i+1)
            ca_dut.w[_i] = 8'sd3;
        ca_dut.b[0] = 8'sd0;
        pulse_and_wait(0);
        for (p=0; p < CA_OUT_SIZE*CA_OUT_SIZE; p=p+1)
            check8(4, "conv_sat", $signed(ca_out[(p*8)+:8]), 8'sd127);

        $display("\nT5: maxpool_layer  4x4 -> 2x2  (expected 5,6,7,9)");
        mb_in[( 0*8)+:8]=8'd1; mb_in[( 1*8)+:8]=8'd3;
        mb_in[( 2*8)+:8]=8'd2; mb_in[( 3*8)+:8]=8'd4;
        mb_in[( 4*8)+:8]=8'd5; mb_in[( 5*8)+:8]=8'd2;
        mb_in[( 6*8)+:8]=8'd1; mb_in[( 7*8)+:8]=8'd6;
        mb_in[( 8*8)+:8]=8'd3; mb_in[( 9*8)+:8]=8'd7;
        mb_in[(10*8)+:8]=8'd8; mb_in[(11*8)+:8]=8'd2;
        mb_in[(12*8)+:8]=8'd4; mb_in[(13*8)+:8]=8'd1;
        mb_in[(14*8)+:8]=8'd3; mb_in[(15*8)+:8]=8'd9;
        pulse_and_wait(1);
        check8(5, "pool(0,0)", $signed(mb_out[(0*8)+:8]), 8'sd5);
        check8(5, "pool(0,1)", $signed(mb_out[(1*8)+:8]), 8'sd6);
        check8(5, "pool(1,0)", $signed(mb_out[(2*8)+:8]), 8'sd7);
        check8(5, "pool(1,1)", $signed(mb_out[(3*8)+:8]), 8'sd9);

        $display("\nT6: fc_layer  USE_RELU=1  n0=10  n1=0");
        fc_in[(0*8)+:8]=8'sd1; fc_in[(1*8)+:8]=8'sd2;
        fc_in[(2*8)+:8]=8'sd3; fc_in[(3*8)+:8]=8'sd4;
        fc_relu_dut.w[0*FC_IN+0]=8'sd1; fc_relu_dut.w[0*FC_IN+1]=8'sd1;
        fc_relu_dut.w[0*FC_IN+2]=8'sd1; fc_relu_dut.w[0*FC_IN+3]=8'sd1;
        fc_relu_dut.b[0]=8'sd0;
        fc_relu_dut.w[1*FC_IN+0]=8'sd1;  fc_relu_dut.w[1*FC_IN+1]=-8'sd1;
        fc_relu_dut.w[1*FC_IN+2]=8'sd1;  fc_relu_dut.w[1*FC_IN+3]=-8'sd1;
        fc_relu_dut.b[1]=8'sd0;
        pulse_and_wait(2);
        check8(6, "fc_relu n0", $signed(fc_out_relu[(0*8)+:8]), 8'sd10);
        check8(6, "fc_relu n1", $signed(fc_out_relu[(1*8)+:8]), 8'sd0);

        $display("\nT7: fc_layer  USE_RELU=0  n0=10  n1=-2");
        fc_noact_dut.w[0*FC_IN+0]=8'sd1; fc_noact_dut.w[0*FC_IN+1]=8'sd1;
        fc_noact_dut.w[0*FC_IN+2]=8'sd1; fc_noact_dut.w[0*FC_IN+3]=8'sd1;
        fc_noact_dut.b[0]=8'sd0;
        fc_noact_dut.w[1*FC_IN+0]=8'sd1;  fc_noact_dut.w[1*FC_IN+1]=-8'sd1;
        fc_noact_dut.w[1*FC_IN+2]=8'sd1;  fc_noact_dut.w[1*FC_IN+3]=-8'sd1;
        fc_noact_dut.b[1]=8'sd0;
        pulse_and_wait(3);
        check8(7, "fc_noact n0", $signed(fc_out_noact[(0*8)+:8]), 8'sd10);
        check8(7, "fc_noact n1", $signed(fc_out_noact[(1*8)+:8]), -8'sd2);

        $display("\nT8: fc_layer  neg saturation  (acc=-1524 -> -128)");
        fe_in[(0*8)+:8]=8'sd127; fe_in[(1*8)+:8]=8'sd127;
        fe_in[(2*8)+:8]=8'sd127; fe_in[(3*8)+:8]=8'sd127;
        fe_dut.w[0]=-8'sd3; fe_dut.w[1]=-8'sd3;
        fe_dut.w[2]=-8'sd3; fe_dut.w[3]=-8'sd3;
        fe_dut.b[0]=8'sd0;
        pulse_and_wait(4);
        check8(8, "fc_neg_sat", $signed(fe_out), -8'sd128);

        $display("");
        $display("======================================================");
        $display("  Phase 1 Results: %0d / %0d unit tests passed",
                 tests_passed, tests_run);
        if (tests_passed == tests_run)
            $display("  ALL UNIT TESTS PASSED -- proceeding to system test");
        else
            $display("  *** UNIT TEST FAILURES -- review above ***");
        $display("======================================================");

        // ==================================================
        // PHASE 2 - FULL SYSTEM TEST
        // ==================================================
        $display("");
        $display("======================================================");
        $display("  PHASE 2: Full System Test  (lenet_top end-to-end)");
        $display("======================================================");

        if (TRAINED_WEIGHTS) begin
            $display("  Mode   : TRAINED weights (Python/MNIST)");
            $display("  Expect : class %0d", EXPECTED_CLASS);
            load_trained_weights;
            load_test_image;
        end else begin
            $display("  Mode   : RANDOM weights");
            $display("  Image  : 32x32 gradient");
            load_random_weights;
            load_gradient_image;
        end

        @(posedge clk);
        @(negedge clk); sys_start = 1;
        @(negedge clk); sys_start = 0;

        $display("\n  Inference running...");
        @(posedge sys_done);
        @(posedge clk);

        $display("");
        $display("  +----------+---------------+");
        $display("  |  Class   |  Logit (int8) |");
        $display("  +----------+---------------+");
        for (i=0; i<10; i=i+1)
            $display("  |    %0d     |     %4d      |",
                     i, $signed(logits[(i*8)+:8]));
        $display("  +----------+---------------+");

        predicted  = 0;
        best_logit = $signed(logits[7:0]);
        for (i=1; i<10; i=i+1) begin
            if ($signed(logits[(i*8)+:8]) > best_logit) begin
                best_logit = $signed(logits[(i*8)+:8]);
                predicted  = i;
            end
        end

        nonzero = 0;
        for (i=0; i<10; i=i+1)
            if (logits[(i*8)+:8] !== 8'd0) nonzero = nonzero + 1;

        $display("");
        $display("  Predicted class : %0d  (logit = %0d)", predicted, best_logit);

        if (TRAINED_WEIGHTS) begin
            if (predicted == EXPECTED_CLASS) begin
                $display("  Classification  : CORRECT");
                sys_pass = 1;
            end else
                $display("  Classification  : WRONG (expected %0d)", EXPECTED_CLASS);
        end else begin
            $display("  Non-zero logits : %0d / 10", nonzero);
            if (nonzero > 0) begin
                $display("  System check    : PASS");
                sys_pass = 1;
            end else
                $display("  System check    : WARN (all zero)");
        end

        $display("");
        $display("======================================================");
        $display("  FINAL SUMMARY");
        $display("  Unit tests  : %0d / %0d passed", tests_passed, tests_run);
        if (TRAINED_WEIGHTS)
            $display("  System test : %0s  (digit %0d -> predicted %0d)",
                     sys_pass ? "PASS" : "FAIL", EXPECTED_CLASS, predicted);
        else
            $display("  System test : %0s  (random weights)",
                     sys_pass ? "PASS" : "WARN");
        $display("======================================================");
        $display("");

        $finish;
    end

    initial begin
        #600_000;
        $display("[%0t ns]  TIMEOUT", $time);
        $finish;
    end

endmodule