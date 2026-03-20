# CS151 — LeNet-5 Hardware Accelerator

## Setup

1. Install the required Python packages:
```
   pip install torch torchvision numpy
```

2. Train the model and generate weights:
```
   python train_lenet5.py
```
   At the end of the output, find the block labeled **"Paste these RSHIFT values into lenet_top"**
   and copy the five `localparam RSHIFT_*` lines into `lenet.v` under the comment
   `// ── RSHIFT values (update from train_lenet5.py output)`.

3. Note the ground-truth digit printed in `weights/test_image_label.txt`.

## Running in EDA Playground

4. Go to [edaplayground.com](https://edaplayground.com) and set the simulator to **Icarus Verilog**.

5. Paste `lenet.v` into the **Design** pane and `tb_lenet.v` into the **Testbench** pane.

6. Click the **+** above the left pane and upload all 12 `.hex` files from the `weights/` folder:
   `c1_w`, `c1_b`, `c3_w`, `c3_b`, `c5_w_0`, `c5_w_1`, `c5_b`, `f6_w`, `f6_b`, `out_w`, `out_b`, `test_image`

7. In `tb_lenet.v`, set:
```verilog
   parameter TRAINED_WEIGHTS = 1;
   parameter EXPECTED_CLASS  = <digit from test_image_label.txt>;
```

8. Click **Run**. The output should show `45 / 45 unit tests passed` and `System test : PASS`.
