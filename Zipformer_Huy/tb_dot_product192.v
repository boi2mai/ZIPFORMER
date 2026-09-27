`timescale 1ns/1ps

module tb_top_mul_blocks;

reg clk;

reg [1535:0] in_flat;
reg [1535:0] weight1_flat;
reg [1535:0] weight2_flat;

wire [31:0] result1;
wire [31:0] result2;

integer i;

reg [7:0] in_arr [0:191];
reg [7:0] w1_arr [0:191];
reg [7:0] w2_arr [0:191];

reg [31:0] golden1;
reg [31:0] golden2;

////////////////////////////////////////////////
// DUT
////////////////////////////////////////////////

top_mul_blocks dut(
    .clk(clk),
    .in_flat(in_flat),
    .weight1_flat(weight1_flat),
    .weight2_flat(weight2_flat),
    .result1(result1),
    .result2(result2)
);

////////////////////////////////////////////////
// CLOCK
////////////////////////////////////////////////

initial begin
    clk = 0;
    forever #5 clk = ~clk;
end

////////////////////////////////////////////////
// TEST
////////////////////////////////////////////////

initial begin

    golden1 = 0;
    golden2 = 0;

    // generate data
    for(i=0;i<192;i=i+1) begin
        in_arr[i] = $urandom%10;
        w1_arr[i] = $urandom%10;
        w2_arr[i] = $urandom%10;
    end

    // pack bus
    for(i=0;i<192;i=i+1) begin
        in_flat[i*8 +: 8]      = in_arr[i];
        weight1_flat[i*8 +: 8] = w1_arr[i];
        weight2_flat[i*8 +: 8] = w2_arr[i];
    end

    // compute golden
    for(i=0;i<192;i=i+1) begin
        golden1 = golden1 + in_arr[i]*w1_arr[i];
        golden2 = golden2 + in_arr[i]*w2_arr[i];
    end

    // chờ pipeline chạy
    repeat(15) @(posedge clk);

    $display("RESULT1 = %d", result1);
    $display("GOLDEN1 = %d", golden1);

    $display("RESULT2 = %d", result2);
    $display("GOLDEN2 = %d", golden2);

    if(result1 == golden1)
        $display("TEST1 PASS");
    else
        $display("TEST1 FAIL");

    if(result2 == golden2)
        $display("TEST2 PASS");
    else
        $display("TEST2 FAIL");

    $finish;

end

endmodule