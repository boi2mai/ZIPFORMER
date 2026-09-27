// processing element

module PE #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH = 32
)(
    input wire clk,
    input wire rst_n,
    input wire en,
    input wire clr_acc,

    input wire signed [DATA_WIDTH-1:0] in_a,
    input wire signed [DATA_WIDTH-1:0] in_b,
    
    output reg signed [DATA_WIDTH-1:0] out_a,
    output reg signed [DATA_WIDTH-1:0] out_b,
    output reg signed [ACC_WIDTH-1:0] acc
    ;
);
    wire signed [2*DATA_WIDTH-1:0] product;
    assign product = in_a*in_b;

    always @(posedge clk or negedge rst_n) begin
        if (rst_n) begin
            out_a <= 0;
            out_b <= 0;
            acc <= 0;
        end
        else if (clr_acc) begin
            acc <= 0;
            out_a <= 0;
            out_b <= 0;
        end
        else if (en) begin
            out_a <= in_a;
            out_b <= in_b;
            acc <= acc + product;
        end
    end
endmodule