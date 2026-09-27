// systolic array

module systolic_array_4x4 #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32
)(
    input clk,
    input rst_n,
    input en,
    input clr_acc,

    input  signed [DATA_WIDTH-1:0] a_in [0:3],
    input  signed [DATA_WIDTH-1:0] b_in [0:3],

    output signed [ACC_WIDTH-1:0] c_out [0:3][0:3]
);

    wire signed [DATA_WIDTH-1:0] a_bus [0:3][0:3];
    wire signed [DATA_WIDTH-1:0] b_bus [0:3][0:3];

    genvar i,j;

    generate
    for(i=0;i<4;i=i+1) begin : ROW
        for(j=0;j<4;j=j+1) begin : COL

            wire signed [DATA_WIDTH-1:0] a_val;
            wire signed [DATA_WIDTH-1:0] b_val;

            assign a_val = (j==0) ? a_in[i] : a_bus[i][j-1];
            assign b_val = (i==0) ? b_in[j] : b_bus[i-1][j];

            PE #(
            .DATA_WIDTH(DATA_WIDTH),
            .ACC_WIDTH(ACC_WIDTH)
            ) PE (
            .clk(clk),
            .rst_n(rst_n),
            .en(en),
            .clr_acc(clr_acc),

            .in_a(a_val),
            .in_b(b_val),

            .out_a(a_bus[i][j]),
            .out_b(b_bus[i][j]),

            .acc(c_out[i][j])
            );
            end
        end
    endgenerate

endmodule