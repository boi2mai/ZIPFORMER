`timescale 1ns / 1ps

module tb_AXI_master;

parameter ADDR_WIDTH = 32;
parameter DATA_WIDTH = 128;

reg clk;
reg reset_n;

// WRITE ADDRESS
wire awvalid;
wire [ADDR_WIDTH-1:0] awaddr;
reg awready;

// WRITE DATA
wire wvalid;
wire [DATA_WIDTH-1:0] wdata;
wire wlast;
reg wready;

// WRITE RESPONSE
reg bvalid;
reg [1:0] bresp;
wire bready;

// READ ADDRESS
wire arvalid;
wire [ADDR_WIDTH-1:0] araddr;
reg arready;

// READ DATA
reg rvalid;
reg [DATA_WIDTH-1:0] rdata;
reg rlast;
reg [1:0] rresp;
wire rready;

// CONTROL
reg start_write;
reg start_read;

reg [ADDR_WIDTH-1:0] addr_write;
reg [ADDR_WIDTH-1:0] addr_read;
reg [DATA_WIDTH-1:0] data_write;

wire [DATA_WIDTH-1:0] data_out;
wire valid;

/////////////////////////////////////////////////////////
// SIMPLE MEMORY MODEL
/////////////////////////////////////////////////////////

reg [DATA_WIDTH-1:0] memory [0:1023];

/////////////////////////////////////////////////////////
// DUT
/////////////////////////////////////////////////////////

AXI_master dut (

.clk(clk),
.reset_n(reset_n),

.awready(awready),
.awlen(8'd0),
.awvalid(awvalid),
.awaddr(awaddr),
.awprot(),
.awsize(),
.awburst(),

.wready(wready),
.wvalid(wvalid),
.wdata(wdata),
.wstrb(),
.wlast(wlast),

.bvalid(bvalid),
.bresp(bresp),
.bready(bready),

.arready(arready),
.arvalid(arvalid),
.araddr(araddr),
.arprot(),
.arlen(),
.arsize(),
.arburst(),

.rvalid(rvalid),
.rdata(rdata),
.rresp(rresp),
.rlast(rlast),
.rready(rready),

.start_write(start_write),
.start_read(start_read),
.addr_read(addr_read),
.addr_write(addr_write),
.data_write(data_write),

.data_out(data_out),
.done(),
.valid(valid),
.status()
);

/////////////////////////////////////////////////////////
// CLOCK
/////////////////////////////////////////////////////////

initial begin
    clk = 0;
    forever #5 clk = ~clk;
end

/////////////////////////////////////////////////////////
// RESET
/////////////////////////////////////////////////////////

initial begin
    reset_n = 0;
    #20;
    reset_n = 1;
end

/////////////////////////////////////////////////////////
// AXI SLAVE MODEL
/////////////////////////////////////////////////////////

integer write_addr_reg;
integer read_addr_reg;

always @(posedge clk) begin

    // default
    awready <= 0;
    wready  <= 0;
    bvalid  <= 0;

    arready <= 0;
    rvalid  <= 0;
    rlast   <= 0;

    ////////////////////////////
    // WRITE ADDRESS HANDSHAKE
    ////////////////////////////

    if (awvalid) begin
        awready <= 1;
        write_addr_reg <= awaddr >> 4;   // 128bit word address
    end

    ////////////////////////////
    // WRITE DATA
    ////////////////////////////

    if (wvalid) begin
        wready <= 1;

        memory[write_addr_reg] <= wdata;

        if (wlast) begin
            bvalid <= 1;
            bresp <= 2'b00;
        end
    end

    ////////////////////////////
    // READ ADDRESS
    ////////////////////////////

    if (arvalid) begin
        arready <= 1;
        read_addr_reg <= araddr >> 4;
    end

    ////////////////////////////
    // READ DATA
    ////////////////////////////

    if (rready && arready) begin
        rvalid <= 1;
        rdata <= memory[read_addr_reg];
        rlast <= 1;
        rresp <= 2'b00;
    end

end

/////////////////////////////////////////////////////////
// TEST SEQUENCE
/////////////////////////////////////////////////////////

initial begin

    start_write = 0;
    start_read = 0;

    addr_write = 0;
    addr_read = 0;
    data_write = 0;

    wait(reset_n);

    #20;

    //////////////////////////////////////////////////////
    // WRITE TEST
    //////////////////////////////////////////////////////

    $display("---- WRITE TEST ----");

    addr_write = 32'h00000040;
    data_write = 128'hDEADBEEF_12345678_ABCDEF11_98765432;

    start_write = 1;
    #10;
    start_write = 0;

    #100;

    if(memory[4] == data_write)
        $display("WRITE SUCCESS");
    else
        $display("WRITE FAIL");

    //////////////////////////////////////////////////////
    // READ TEST
    //////////////////////////////////////////////////////

    $display("---- READ TEST ----");

    addr_read = 32'h00000040;

    start_read = 1;
    #10;
    start_read = 0;

    #100;

    if(data_out == data_write)
        $display("READ SUCCESS");
    else
        $display("READ FAIL");

    //////////////////////////////////////////////////////
    // FINISH
    //////////////////////////////////////////////////////

    #50;
    $display("SIMULATION FINISHED");
    $stop;

end

endmodule