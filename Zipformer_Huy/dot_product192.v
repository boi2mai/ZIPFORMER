
// processing element
module pe_mul(
    input clk,
    input [7:0] in,
    input [7:0] weight,
    output reg [15:0] out
);

always @(posedge clk)
    out <= in * weight;

endmodule

// mac_block
module mac_block #(
    parameter N = 256
)(
    input clk,

    input  [8*N-1:0]  in_flat,
    input  [8*N-1:0]  weight_flat,

    output [31:0] out
);

wire [7:0] in     [0:N-1];
wire [7:0] weight [0:N-1];

wire [15:0] mul_out [0:N-1];
wire [16*N-1:0] mul_flat;

genvar i;

// unpack input
generate
for(i=0;i<N;i=i+1) begin: UNPACK
    assign in[i]     = in_flat[i*8 +: 8];
    assign weight[i] = weight_flat[i*8 +: 8];
end
endgenerate

// multiplier array
generate
for(i=0;i<N;i=i+1) begin: MUL_ARRAY
    pe_mul pe(
        .clk(clk),
        .in(in[i]),
        .weight(weight[i]),
        .out(mul_out[i])
    );
end
endgenerate

// pack
generate
for(i=0;i<N;i=i+1) begin: PACK
    assign mul_flat[i*16 +: 16] = mul_out[i];
end
endgenerate

adder_tree #(N) adder_tree_inst(
    .clk(clk),
    .in_flat(mul_flat),
    .sum(out)
);

endmodule

//adder_tree
module adder_tree #(
    parameter N = 256
)(
    input clk,
    input [16*N-1:0] in_flat,
    output reg [31:0] sum
);

wire [15:0] in [0:N-1];

genvar k;

generate
for(k=0;k<N;k=k+1) begin: UNPACK
    assign in[k] = in_flat[k*16 +: 16];
end
endgenerate

reg [16:0] s1 [0:N/2-1];
reg [17:0] s2 [0:N/4-1];
reg [18:0] s3 [0:N/8-1];
reg [19:0] s4 [0:N/16-1];
reg [20:0] s5 [0:N/32-1];
reg [21:0] s6 [0:N/64-1];
reg [22:0] s7 [0:N/128-1];
reg [23:0] s8 [0:N/256-1];

integer i;

always @(posedge clk) begin

for(i=0;i<N/2;i=i+1)
    s1[i] <= in[2*i] + in[2*i+1];

for(i=0;i<N/4;i=i+1)
    s2[i] <= s1[2*i] + s1[2*i+1];

for(i=0;i<N/8;i=i+1)
    s3[i] <= s2[2*i] + s2[2*i+1];

for(i=0;i<N/16;i=i+1)
    s4[i] <= s3[2*i] + s3[2*i+1];

for(i=0;i<N/32;i=i+1)
    s5[i] <= s4[2*i] + s4[2*i+1];

for(i=0;i<N/64;i=i+1)
    s6[i] <= s5[2*i] + s5[2*i+1];

for(i=0;i<N/128;i=i+1)
    s7[i] <= s6[2*i] + s6[2*i+1];

for(i=0;i<N/256;i=i+1)
    s8[i] <= s7[2*i] + s7[2*i+1];

sum <= s8[0];

end

endmodule

module top_mul_blocks (

    input clk,

    input [2047:0] in_flat,
    input [2047:0] weight1_flat,
    input [2047:0] weight2_flat,

    output [31:0] result1,
    output [31:0] result2
);

mac_block #(256) block1(
    .clk(clk),
    .in_flat(in_flat),
    .weight_flat(weight1_flat),
    .out(result1)
);

mac_block #(256) block2(
    .clk(clk),
    .in_flat(in_flat),
    .weight_flat(weight2_flat),
    .out(result2)
);

endmodule
